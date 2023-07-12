import sys
from PIL import Image
import torch
from kandinsky2.model.model_creation import create_model, create_gaussian_diffusion
from kandinsky2.train_utils.train_module_pl2_1 import Decoder
import argparse
import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from kandinsky2.train_utils.data.dataset_unclip_2_1 import create_loader
from kandinsky2.train_utils.utils import freeze_decoder

from kandinsky2.model.text_encoders import TextEncoder
from kandinsky2.model.utils import get_obj_from_str
from kandinsky2.vqgan.autoencoder import VQModelInterface, AutoencoderKL, MOVQ
from kandinsky2.train_utils.trainer_2_1_uclip import train_unclip
from kandinsky2.model.resample import UniformSampler
from omegaconf import OmegaConf
import clip
import argparse

def drop_first_layer(path):
    d = {}
    state_dict = torch.load(path)
    for key in state_dict.keys():
        if key != 'input_blocks.0.0.weight':
            d[key] = state_dict[key]
    return d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    device = config['device']
    model = create_model(**config['model_config'])
    diffusion = create_gaussian_diffusion(**config['diffusion_config'])
    print('start loading')
    if config['params_path'] != '':
        if config['drop_first_layer']:
            model.load_state_dict(drop_first_layer(config['params_path']), strict=False)
        else:
            model.load_state_dict(torch.load(config['params_path']))
    model = freeze_decoder(model, **config['freeze']).to(device)
    train_loader = create_loader(**config['data']['train'])
    image_encoder = MOVQ(**config['image_enc_params']["params"]).half()
    image_encoder.load_state_dict(torch.load(config['image_enc_params']["ckpt_path"]))
    image_encoder = image_encoder.eval().to(device)
    schedule_sampler = UniformSampler(diffusion)
    text_encoder = TextEncoder(**config['text_enc_params']).eval().half().to(device)
    optimizer = get_obj_from_str(config['optim_params']["name"])(
            model.parameters(), **config['optim_params']["params"]
        )
    if 'scheduler_params' in config:
        lr_scheduler = get_obj_from_str(config['scheduler_params']["name"])(
                optimizer, **config['scheduler_params']["params"]
            )
    else:
        lr_scheduler = None
    clip_model, _ = clip.load(config['clip_name'], device="cpu", jit=False)
    clip_model.transformer = None
    clip_model.positional_embedding = None
    clip_model.ln_final = None
    clip_model.token_embedding = None
    clip_model.text_projection = None
    clip_model = clip_model.eval().to(device)
    train_unclip(unet=model, diffusion=diffusion, image_encoder=image_encoder,
                  clip_model=clip_model, text_encoder=text_encoder, optimizer=optimizer,
                  lr_scheduler=lr_scheduler, schedule_sampler=schedule_sampler, 
                  train_loader=train_loader, val_loader=None, scale=config['image_enc_params']['scale'],
                  num_epochs=config['num_epochs'], save_every=config['save_every'], save_name=config['save_name'],
                  save_path=config['save_path'],  inpainting=config['inpainting'], device=device)
if __name__ == '__main__':
    main()
