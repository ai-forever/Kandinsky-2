import sys

import copy
import functools
import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from ..model.resample import UniformSampler
from ..vqgan.autoencoder import VQModelInterface, AutoencoderKL, MOVQ
from ..model.text_encoders import TextEncoder
from ..model.utils import get_obj_from_str
from .ema import EMA
from .utils import generate_mask, get_image_mask
import clip


class Decoder(pl.LightningModule):
    def __init__(
        self,
        unet,
        diffusion,
        ema_rate,
        optim_params,
        scheduler_params,
        image_enc_params,
        text_enc_params,
        clip_name,
        use_ema=False,
        inpainting=False,
    ):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.image_enc_params = image_enc_params
        self.text_enc_params = text_enc_params
        self.ema_rate = ema_rate
        self.use_ema = use_ema
        self.schedule_sampler = UniformSampler(diffusion)
        self.inpainting = inpainting

        self.create_image_encoder()
        self.create_text_encoder()

        self.optim_params = optim_params
        self.scheduler_params = scheduler_params
        if use_ema:
            self.ema_params = EMA(
                self.unet,
                ema_rate,
            )

        self.clip_model, _ = clip.load(clip_name, device="cpu", jit=False)
        self.clip_model.transformer = None
        self.clip_model.positional_embedding = None
        self.clip_model.ln_final = None
        self.clip_model.token_embedding = None
        self.clip_model.text_projection = None

    def create_image_encoder(
        self,
    ):
        if self.image_enc_params is not None:
            self.use_image_enc = True
            self.scale = self.image_enc_params["scale"]
            self.image_enc_name = self.image_enc_params["name"]
            if self.image_enc_params["name"] == "AutoencoderKL":
                self.image_encoder = AutoencoderKL(**self.image_enc_params["params"])
            elif self.image_enc_params["name"] == "VQModelInterface":
                self.image_encoder = VQModelInterface(**self.image_enc_params["params"])
            elif self.image_enc_params["name"] == "MOVQ":
                self.image_encoder = MOVQ(**self.image_enc_params["params"])
                self.image_encoder.load_state_dict(
                    torch.load(self.image_enc_params["ckpt_path"])
                )
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            self.use_image_enc = False

    def create_text_encoder(
        self,
    ):
        if self.text_enc_params is not None:
            self.use_text_enc = True
            self.text_encoder = TextEncoder(**self.text_enc_params).eval().half()
        else:
            self.use_text_enc = False

    def configure_optimizers(self):
        optimizer = get_obj_from_str(self.optim_params["name"])(
            self.unet.parameters(), **self.optim_params["params"]
        )
        lr_scheduler = get_obj_from_str(self.scheduler_params["name"])(
            optimizer, **self.scheduler_params["params"]
        )
        return [optimizer], {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

    def prepare_image(self, batch):
        if self.use_image_enc:
            with torch.no_grad():
                if self.image_enc_name == "AutoencoderKL":
                    batch = self.image_encoder.encode(batch).sample()
                elif self.image_enc_name == "VQModelInterface":
                    batch = self.image_encoder.encode(batch)
                elif self.image_enc_name == "MOVQ":
                    batch = self.image_encoder.encode(batch)
                batch = batch * self.scale
        return batch

    def prepare_cond(self, cond):
        if self.use_text_enc:
            mask = None
            new_cond = {}
            for key in cond.keys():
                if key not in ["tokens", "mask", "clip_image"]:
                    new_cond[key] = cond[key]
            if "mask" in cond:
                mask = cond["mask"]
            with torch.no_grad():
                new_cond["image_emb"] = self.clip_model.encode_image(
                    cond["clip_image"]
                ).float()
            with torch.no_grad():
                new_cond["full_emb"], new_cond["pooled_emb"] = self.text_encoder(
                    cond["tokens"].long(), mask
                )
            del cond
            return new_cond
        return cond

    def model_step(self, batch, stage):
        image, cond = batch
        image = self.prepare_image(image)

        if self.inpainting:
            image_mask = get_image_mask(image.shape[0], image.shape[-2:])
            image_mask = image_mask.to(image.device).unsqueeze(1).to(image.dtype)
            # image_mask = 1. - image_mask
            cond["inpaint_image"] = image * image_mask
            cond["inpaint_mask"] = image_mask

        cond = self.prepare_cond(cond)
        t, weights = self.schedule_sampler.sample(image.shape[0], image.device)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.unet,
            image,
            t,
            model_kwargs=cond,
        )
        losses = compute_losses()
        loss = losses["loss"].mean()
        self.log(f"{stage}_loss", loss.detach().cpu().item(), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, "valid")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_params(self.unet)
