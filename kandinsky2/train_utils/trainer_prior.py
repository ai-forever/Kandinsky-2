import sys

import copy
import functools
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from .utils import generate_mask, get_image_mask
import clip

def encode_text(tok, clip_model):
    with torch.no_grad():
        x = clip_model.token_embedding(tok).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        txt_feat_seq = x
        txt_feat = x[torch.arange(x.shape[0]), tok.argmax(dim=-1)] @ clip_model.text_projection
        txt_feat, txt_feat_seq = txt_feat.float(), txt_feat_seq.float()
        return txt_feat, txt_feat_seq

def encode_image(image, clip_model, clip_mean, clip_std):
    with torch.no_grad():
        return (clip_model.encode_image(image).float() - clip_mean) / clip_std

def train_prior(model, diffusion,
                  clip_model, optimizer,
                  lr_scheduler=None, schedule_sampler=None, 
                  train_loader=None, val_loader=None,
                  num_epochs=2, save_every=1000, save_name='model',
                  save_path='', device='cuda:0'):
    train_step = 0
    for epoch in range(num_epochs):
        progress = tqdm(total=len(train_loader), desc='finetuning goes brrr')
        for batch in train_loader:
            optimizer.zero_grad()
            image, cond = batch
            image = image.to(device)
            for key in cond.keys():
                cond[key] = cond[key].to(device)
            image = encode_image(image, clip_model, model.clip_mean.to(device), model.clip_std.to(device))
            txt_feat, txt_feat_seq = encode_text(cond['tokens'], clip_model)
            cond = {
            "text_emb": txt_feat,
            "text_enc": txt_feat_seq,
            "mask": cond['mask'],
            "causal_mask": model.causal_mask,
            }
            t, weights = schedule_sampler.sample(image.shape[0], image.device)
            compute_losses = functools.partial(
                    diffusion.training_losses,
                    model.model,
                    image,
                    t,
                    model_kwargs=cond,
                )
            losses = compute_losses()
            loss = losses["loss"].mean()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            train_step += 1
            if train_step % save_every == 0:
                torch.save(model.state_dict(), os.path.join(save_path, save_name + str(train_step) + '.ckpt'))
            progress.update()
            progress.set_postfix({"loss": loss.item()})
        