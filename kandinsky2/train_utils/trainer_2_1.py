import sys

import copy
import functools
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from ..vqgan.autoencoder import VQModelInterface, AutoencoderKL, MOVQ
from .utils import generate_mask, get_image_mask
import clip

def prepare_image(batch, image_encoder, scale=1):
    with torch.no_grad():
        if type(image_enc_name) == AutoencoderKL:
            batch = image_encoder.encode(batch).sample()
        elif type(image_enc_name) == VQModelInterface:
            batch = image_encoder.encode(batch)
        elif type(image_enc_name) == MOVQ:
            batch = image_encoder.encode(batch)
        batch = batch * scale
    return batch

def prepare_cond(cond, text_encoder, clip_model):
    mask = None
    new_cond = {}
    for key in cond.keys():
        if key not in ['tokens', 'mask', 'clip_image']:
            new_cond[key] = cond[key]
    if 'mask' in cond:
        mask = cond['mask']
    with torch.no_grad():
        new_cond['image_emb'] = clip_model.encode_image(cond['clip_image']).float()
    with torch.no_grad():
        new_cond['full_emb'], new_cond['pooled_emb'] = text_encoder(
                    cond['tokens'].long(), mask)
    del cond
    return new_cond

def train_decoder(unet, diffusion, image_encoder,
                  clip_model, text_encoder, optimizer,
                  lr_scheduler, schedule_sampler, 
                  train_loader, val_loader=None, scale=1,
                  num_epochs=2, save_every=1000, save_name='model',
                  save_path='',  inpainting=False, deivce='cuda:0'):
    
    train_step = 0
    for epoch in tqdm(num_epochs):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            image, cond = batch
            image = image.to(device)
            for key in cond.keys():
                cond[key] = cond[key].to(device)
            if inpainting:
                image_mask = get_image_mask(image.shape[0], image.shape[-2:])
                image_mask = image_mask.to(image.device).unsqueeze(1).to(image.dtype)
                image_mask = 1. - image_mask
                cond['inpaint_image'] = image * image_mask
                cond['inpaint_mask'] = image_mask
                
            train_step += 1
            if train_step % save_every == 0:
                
        