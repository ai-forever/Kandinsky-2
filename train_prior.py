import sys
from PIL import Image
import torch
from kandinsky2.model.model_creation import create_model, create_gaussian_diffusion
from kandinsky2.train_utils.train_module_pl2_1 import Decoder
import argparse
import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from kandinsky2.train_utils.data.dataset_prior import create_loader

from kandinsky2.model.utils import get_obj_from_str
from kandinsky2.train_utils.trainer_prior import train_prior
from kandinsky2.model.resample import UniformSampler
from kandinsky2.model.prior import PriorDiffusionModel, CustomizedTokenizer
import argparse
from omegaconf import OmegaConf
import clip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    device = config['device']
    clip_mean, clip_std = torch.load(
            config["clip_mean_std_path"], map_location="cpu"
        )
    tokenizer = CustomizedTokenizer()
    model = PriorDiffusionModel(
            config['model_config'],
            tokenizer,
            clip_mean,
            clip_std,
        )
    diffusion = model.create_prior_diffusion()
    print('start loading')
    if config['params_path'] != '':
        model.load_state_dict(torch.load(config['params_path']))
    model = model.to(device)
    train_loader = create_loader(**config['data']['train'])
    schedule_sampler = UniformSampler(diffusion)
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
    clip_model = clip_model.eval().to(device)
    train_prior(model=model, diffusion=diffusion,
                  clip_model=clip_model, optimizer=optimizer,
                  lr_scheduler=lr_scheduler, schedule_sampler=schedule_sampler, 
                  train_loader=train_loader, val_loader=None,
                  num_epochs=config['num_epochs'], save_every=config['save_every'], save_name=config['save_name'],
                  save_path=config['save_path'],  device=device)
if __name__ == '__main__':
    main()
