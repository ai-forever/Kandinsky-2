from transformers import AutoTokenizer
from PIL import Image
import cv2
import torch
from omegaconf import OmegaConf
import math
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import clip
from transformers import AutoTokenizer

from .model.text_encoders import TextEncoder
from .vqgan.autoencoder import VQModelInterface, AutoencoderKL, MOVQ
from .model.samplers import DDIMSampler, PLMSSampler
from .model.model_creation import create_model, create_gaussian_diffusion
from .model.prior import PriorDiffusionModel, CustomizedTokenizer
from .utils import prepare_image, q_sample, process_images, prepare_mask


class Kandinsky2_1:
    
    def __init__(
        self, 
        config, 
        model_path, 
        prior_path, 
        device, 
        task_type="text2img"
    ):
        self.config = config
        self.device = device
        if device != "cuda":
            self.config["model_config"]["use_fp16"] = False
        self.use_fp16 = self.config["model_config"]["use_fp16"]
        self.task_type = task_type
        self.clip_image_size = config["clip_image_size"]
        if task_type == "text2img":
            self.config["model_config"]["up"] = False
            self.config["model_config"]["inpainting"] = False
        elif task_type == "inpainting":
            self.config["model_config"]["up"] = False
            self.config["model_config"]["inpainting"] = True
        else:
            raise ValueError("Only text2img and inpainting is available")

        self.tokenizer1 = AutoTokenizer.from_pretrained(self.config["tokenizer_name"])
        self.tokenizer2 = CustomizedTokenizer()
        clip_mean, clip_std = torch.load(
            config["prior"]["clip_mean_std_path"], map_location="cpu"
        )

        self.prior = PriorDiffusionModel(
            config["prior"]["params"],
            self.tokenizer2,
            clip_mean,
            clip_std,
        )
        self.prior.load_state_dict(torch.load(prior_path, map_location='cpu'), strict=False)
        if self.use_fp16:
            self.prior = self.prior.half()
        self.text_encoder = TextEncoder(**self.config["text_enc_params"])
        if self.use_fp16:
            self.text_encoder = self.text_encoder.half()

        self.clip_model, self.preprocess = clip.load(
            config["clip_name"], device=self.device, jit=False
        )
        self.clip_model.eval()

        if self.config["image_enc_params"] is not None:
            self.use_image_enc = True
            self.scale = self.config["image_enc_params"]["scale"]
            if self.config["image_enc_params"]["name"] == "AutoencoderKL":
                self.image_encoder = AutoencoderKL(
                    **self.config["image_enc_params"]["params"]
                )
            elif self.config["image_enc_params"]["name"] == "VQModelInterface":
                self.image_encoder = VQModelInterface(
                    **self.config["image_enc_params"]["params"]
                )
            elif self.config["image_enc_params"]["name"] == "MOVQ":
                self.image_encoder = MOVQ(**self.config["image_enc_params"]["params"])
                self.image_encoder.load_state_dict(
                    torch.load(self.config["image_enc_params"]["ckpt_path"])
                )
            self.image_encoder.eval()
        else:
            self.use_image_enc = False
            
        self.config["model_config"]["cache_text_emb"] = True
        self.model = create_model(**self.config["model_config"])
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        if self.use_fp16:
            self.model.convert_to_fp16()
            self.image_encoder = self.image_encoder.half()

            self.model_dtype = torch.float16
        else:
            self.model_dtype = torch.float32
            
        self.image_encoder = self.image_encoder.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.prior = self.prior.to(self.device).eval()
        self.model.eval()
        self.model.to(self.device)

    def get_new_h_w(self, h, w):
        new_h = h // 64
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        return new_h * 8, new_w * 8

    @torch.no_grad()
    def encode_text(self, text_encoder, tokenizer, prompt, batch_size):
        text_encoding = tokenizer(
            [prompt] * batch_size + [""] * batch_size,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        tokens = text_encoding["input_ids"].to(self.device)
        mask = text_encoding["attention_mask"].to(self.device)

        full_emb, pooled_emb = text_encoder(tokens=tokens, mask=mask)
        return full_emb, pooled_emb

    @torch.no_grad()
    def generate_clip_emb(
        self,
        prompt,
        batch_size=1,
        prior_cf_scale=4,
        prior_steps="25",
        negative_prior_prompt="",
    ):
        prompts_batch = [prompt for _ in range(batch_size)]
        prior_cf_scales_batch = [prior_cf_scale] * len(prompts_batch)
        prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device=self.device)
        max_txt_length = self.prior.model.text_ctx
        tok, mask = self.tokenizer2.padded_tokens_and_mask(
            prompts_batch, max_txt_length
        )
        cf_token, cf_mask = self.tokenizer2.padded_tokens_and_mask(
            [negative_prior_prompt], max_txt_length
        )
        if not (cf_token.shape == tok.shape):
            cf_token = cf_token.expand(tok.shape[0], -1)
            cf_mask = cf_mask.expand(tok.shape[0], -1)
        tok = torch.cat([tok, cf_token], dim=0)
        mask = torch.cat([mask, cf_mask], dim=0)
        tok, mask = tok.to(device=self.device), mask.to(device=self.device)

        x = self.clip_model.token_embedding(tok).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND|
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        txt_feat_seq = x
        txt_feat = (x[torch.arange(x.shape[0]), tok.argmax(dim=-1)] @ self.clip_model.text_projection)
        txt_feat, txt_feat_seq = txt_feat.float().to(self.device), txt_feat_seq.float().to(self.device)
        img_feat = self.prior(
            txt_feat,
            txt_feat_seq,
            mask,
            prior_cf_scales_batch,
            timestep_respacing=prior_steps,
        )
        return img_feat.to(self.model_dtype)

    @torch.no_grad()
    def encode_images(self, image, is_pil=False):
        if is_pil:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.clip_model.encode_image(image).to(self.model_dtype)

    @torch.no_grad()
    def generate_img(
        self,
        prompt,
        img_prompt,
        batch_size=1,
        diffusion=None,
        guidance_scale=7,
        init_step=None,
        noise=None,
        init_img=None,
        img_mask=None,
        h=512,
        w=512,
        sampler="ddim_sampler",
        num_steps=50,
    ):
        new_h, new_w = self.get_new_h_w(h, w)
        full_batch_size = batch_size * 2
        model_kwargs = {}

        if init_img is not None and self.use_fp16:
            init_img = init_img.half()
        if img_mask is not None and self.use_fp16:
            img_mask = img_mask.half()
        model_kwargs["full_emb"], model_kwargs["pooled_emb"] = self.encode_text(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer1,
            prompt=prompt,
            batch_size=batch_size,
        )
        model_kwargs["image_emb"] = img_prompt

        if self.task_type == "inpainting":
            init_img = init_img.to(self.device)
            img_mask = img_mask.to(self.device)
            model_kwargs["inpaint_image"] = init_img * img_mask
            model_kwargs["inpaint_mask"] = img_mask

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :4], model_out[:, 4:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            if sampler == "p_sampler":
                return torch.cat([eps, rest], dim=1)
            else:
                return eps

        if noise is not None:
            noise = noise.float()
        if self.task_type == "inpainting":
            def denoised_fun(x_start):
                x_start = x_start.clamp(-2, 2)
                return x_start * (1 - img_mask) + init_img * img_mask
        else:
            def denoised_fun(x):
                return x.clamp(-2, 2)

        if sampler == "p_sampler":
            self.model.del_cache()
            samples = diffusion.p_sample_loop(
                model_fn,
                (full_batch_size, 4, new_h, new_w),
                device=self.device,
                noise=noise,
                progress=True,
                model_kwargs=model_kwargs,
                init_step=init_step,
                denoised_fn=denoised_fun,
            )[:batch_size]
            self.model.del_cache()
        else:
            if sampler == "ddim_sampler":
                sampler = DDIMSampler(
                    model=model_fn,
                    old_diffusion=diffusion,
                    schedule="linear",
                )
            elif sampler == "plms_sampler":
                sampler = PLMSSampler(
                    model=model_fn,
                    old_diffusion=diffusion,
                    schedule="linear",
                )
            else:
                raise ValueError("Only ddim_sampler and plms_sampler is available")
                
            self.model.del_cache()
            samples, _ = sampler.sample(
                num_steps,
                batch_size * 2,
                (4, new_h, new_w),
                conditioning=model_kwargs,
                x_T=noise,
                init_step=init_step,
            )
            self.model.del_cache()
            samples = samples[:batch_size]
            
        if self.use_image_enc:
            if self.use_fp16:
                samples = samples.half()
            samples = self.image_encoder.decode(samples / self.scale)
            
        samples = samples[:, :, :h, :w]
        return process_images(samples)

    @torch.no_grad()
    def create_zero_img_emb(self, batch_size):
        img = torch.zeros(1, 3, self.clip_image_size, self.clip_image_size).to(self.device)
        return self.encode_images(img, is_pil=False).repeat(batch_size, 1)

    @torch.no_grad()
    def generate_text2img(
        self,
        prompt,
        num_steps=100,
        batch_size=1,
        guidance_scale=7,
        h=512,
        w=512,
        sampler="ddim_sampler",
        prior_cf_scale=4,
        prior_steps="25",
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        # generate clip embeddings
        image_emb = self.generate_clip_emb(
            prompt,
            batch_size=batch_size,
            prior_cf_scale=prior_cf_scale,
            prior_steps=prior_steps,
            negative_prior_prompt=negative_prior_prompt,
        )
        if negative_decoder_prompt == "":
            zero_image_emb = self.create_zero_img_emb(batch_size=batch_size)
        else:
            zero_image_emb = self.generate_clip_emb(
                negative_decoder_prompt,
                batch_size=batch_size,
                prior_cf_scale=prior_cf_scale,
                prior_steps=prior_steps,
                negative_prior_prompt=negative_prior_prompt,
            )

        image_emb = torch.cat([image_emb, zero_image_emb], dim=0).to(self.device)
        
        # load diffusion
        config = deepcopy(self.config)
        if sampler == "p_sampler":
            config["diffusion_config"]["timestep_respacing"] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config["diffusion_config"])
        
        return self.generate_img(
            prompt=prompt,
            img_prompt=image_emb,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=h,
            w=w,
            sampler=sampler,
            num_steps=num_steps,
            diffusion=diffusion,
        )

    @torch.no_grad()
    def mix_images(
        self,
        images_texts,
        weights,
        num_steps=100,
        batch_size=1,
        guidance_scale=7,
        h=512,
        w=512,
        sampler="ddim_sampler",
        prior_cf_scale=4,
        prior_steps="25",
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        assert len(images_texts) == len(weights) and len(images_texts) > 0
        
        # generate clip embeddings
        image_emb = None
        for i in range(len(images_texts)):
            if image_emb is None:
                if type(images_texts[i]) == str:
                    image_emb = weights[i] * self.generate_clip_emb(
                        images_texts[i],
                        batch_size=1,
                        prior_cf_scale=prior_cf_scale,
                        prior_steps=prior_steps,
                        negative_prior_prompt=negative_prior_prompt,
                    )
                else:
                    image_emb = self.encode_images(images_texts[i], is_pil=True) * weights[i]
            else:
                if type(images_texts[i]) == str:
                    image_emb = image_emb + weights[i] * self.generate_clip_emb(
                        images_texts[i],
                        batch_size=1,
                        prior_cf_scale=prior_cf_scale,
                        prior_steps=prior_steps,
                        negative_prior_prompt=negative_prior_prompt,
                    )
                else:
                    image_emb = image_emb + self.encode_images(images_texts[i], is_pil=True) * weights[i]
                    
        image_emb = image_emb.repeat(batch_size, 1)
        if negative_decoder_prompt == "":
            zero_image_emb = self.create_zero_img_emb(batch_size=batch_size)
        else:
            zero_image_emb = self.generate_clip_emb(
                negative_decoder_prompt,
                batch_size=batch_size,
                prior_cf_scale=prior_cf_scale,
                prior_steps=prior_steps,
                negative_prior_prompt=negative_prior_prompt,
            )
        image_emb = torch.cat([image_emb, zero_image_emb], dim=0).to(self.device)
        
        # load diffusion
        config = deepcopy(self.config)
        if sampler == "p_sampler":
            config["diffusion_config"]["timestep_respacing"] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config["diffusion_config"])
        return self.generate_img(
            prompt="",
            img_prompt=image_emb,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=h,
            w=w,
            sampler=sampler,
            num_steps=num_steps,
            diffusion=diffusion,
        )

    @torch.no_grad()
    def generate_img2img(
        self,
        prompt,
        pil_img,
        strength=0.7,
        num_steps=100,
        batch_size=1,
        guidance_scale=7,
        h=512,
        w=512,
        sampler="ddim_sampler",
        prior_cf_scale=4,
        prior_steps="25",
    ):
        # generate clip embeddings
        image_emb = self.generate_clip_emb(
            prompt,
            batch_size=batch_size,
            prior_cf_scale=prior_cf_scale,
            prior_steps=prior_steps,
        )
        zero_image_emb = self.create_zero_img_emb(batch_size=batch_size)
        image_emb = torch.cat([image_emb, zero_image_emb], dim=0).to(self.device)
        
        # load diffusion
        config = deepcopy(self.config)
        if sampler == "p_sampler":
            config["diffusion_config"]["timestep_respacing"] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config["diffusion_config"])
        
        image = prepare_image(pil_img, h=h, w=w).to(self.device)
        if self.use_fp16:
            image = image.half()
        image = self.image_encoder.encode(image) * self.scale
        
        start_step = int(diffusion.num_timesteps * (1 - strength))
        image = q_sample(
            image,
            torch.tensor(diffusion.timestep_map[start_step - 1]).to(self.device),
            schedule_name=config["diffusion_config"]["noise_schedule"],
            num_steps=config["diffusion_config"]["steps"],
        )
        
        image = image.repeat(2, 1, 1, 1)
        return self.generate_img(
            prompt=prompt,
            img_prompt=image_emb,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=h,
            w=w,
            sampler=sampler,
            num_steps=num_steps,
            diffusion=diffusion,
            noise=image,
            init_step=start_step,
        )

    @torch.no_grad()
    def generate_inpainting(
        self,
        prompt,
        pil_img,
        img_mask,
        num_steps=100,
        batch_size=1,
        guidance_scale=7,
        h=512,
        w=512,
        sampler="ddim_sampler",
        prior_cf_scale=4,
        prior_steps="25",
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        # generate clip embeddings
        image_emb = self.generate_clip_emb(
            prompt,
            batch_size=batch_size,
            prior_cf_scale=prior_cf_scale,
            prior_steps=prior_steps,
            negative_prior_prompt=negative_prior_prompt,
        )
        zero_image_emb = self.create_zero_img_emb(batch_size=batch_size)
        image_emb = torch.cat([image_emb, zero_image_emb], dim=0).to(self.device)
        
        # load diffusion
        config = deepcopy(self.config)
        if sampler == "p_sampler":
            config["diffusion_config"]["timestep_respacing"] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config["diffusion_config"])
        image = prepare_image(pil_img, w, h).to(self.device)
        if self.use_fp16:
            image = image.half()
        image = self.image_encoder.encode(image) * self.scale
        image_shape = tuple(image.shape[-2:])
        img_mask = torch.from_numpy(img_mask).unsqueeze(0).unsqueeze(0)
        img_mask = F.interpolate(
            img_mask,
            image_shape,
            mode="nearest",
        )
        img_mask = prepare_mask(img_mask).to(self.device)
        if self.use_fp16:
            img_mask = img_mask.half()
        image = image.repeat(2, 1, 1, 1)
        img_mask = img_mask.repeat(2, 1, 1, 1)
        
        return self.generate_img(
            prompt=prompt,
            img_prompt=image_emb,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=h,
            w=w,
            sampler=sampler,
            num_steps=num_steps,
            diffusion=diffusion,
            init_img=image,
            img_mask=img_mask,
        )
