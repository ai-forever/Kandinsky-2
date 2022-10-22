from natalle.model_creation import create_model, create_gaussian_diffusion
from transformers import AutoTokenizer
from PIL import Image
import cv2
import torch
from omegaconf import OmegaConf
import clip
import math
from imagen_pytorch.text_encoders import TextEncoder
from imagen_pytorch.vqgan.autoencoder import VQModelInterface, AutoencoderKL
from copy import deepcopy
import torch.nn.functional as F

from .utils import prepare_image, q_sample, process_images

class Natalle:
    def __init__(self, config_path, model_path, device, task_type='text2img', vae_path=None):
        self.config = dict(OmegaConf.load(config_path))
        self.device = device
        self.task_type = task_type
        if task_type == 'text2img' or task_type == 'img2img':
            self.config['model_config']['up'] = False
            self.config['model_config']['inpainting'] = False
        elif task_type == 'inpainting':
            self.config['model_config']['up'] = False
            self.config['model_config']['inpainting'] = True
        else:
            raise ValueError('Only text2img, img2img and inpainting available')
        self.tokenizer1 = AutoTokenizer.from_pretrained(self.config['tokenizer_name1'])
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.config['tokenizer_name2'])
        self.text_encoder = TextEncoder(**self.config['text_enc_params']).to('cuda').eval()

        if vae_path is not None:
            self.config['image_enc_params']['params']['ckpt_path'] = vae_path

        if self.config['image_enc_params'] is not None:
            self.use_image_enc = True
            self.scale = self.config['image_enc_params']['scale']
            if self.config['image_enc_params']['name'] == 'AutoencoderKL':
                self.image_encoder = AutoencoderKL(**self.config['image_enc_params']['params']).to('cuda')
            elif self.config['image_enc_params']['name'] == 'VQModelInterface':
                self.image_encoder = VQModelInterface(**self.config['image_enc_params']['params']).to('cuda')
            self.image_encoder.eval()
        else:
            self.use_image_enc = False
        self.config['model_config']['cache_text_emb'] = True
        self.model = create_model(**self.config['model_config'])
        self.model.load_state_dict(torch.load(model_path), strict=False)
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
    def generate_img(self, prompt, batch_size=1,
                     diffusion=None,
                     guidance_scale=7, progress=True, dynamic_threshold_v=99.5,
                     denoised_type='dynamic_threshold', init_step=None, noise=None,
                     init_img=None, img_mask=None, h=512, w=512,
                     ):
        new_h, new_w = self.get_new_h_w(h, w)
        full_batch_size = batch_size * 2
        text_encoding = self.tokenizer1(
            [prompt] * batch_size + [''] * batch_size,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        tokens = text_encoding['input_ids'].to(self.device)
        mask = text_encoding['attention_mask'].to(self.device)

        model_kwargs = {}
        model_kwargs['full_emb'], model_kwargs['pooled_emb'] = self.text_encoder(tokens=tokens, mask=mask)
        text_encoding2 = self.tokenizer2(
            [prompt] * batch_size + [''] * batch_size,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")
        model_kwargs['input_ids'], model_kwargs['attention_mask'] = text_encoding2['input_ids'].to(self.device), \
                                                                    text_encoding2['attention_mask'].to(self.device)
        if self.task_type == 'inpainting':
            init_img = init_img.to(self.device)
            img_mask = img_mask.to(self.device)
            model_kwargs['inpaint_image'] = (init_img * img_mask)
            model_kwargs['inpaint_mask'] = img_mask

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :4], model_out[:, 4:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        def denoised_fn(x_start,):
            return (
                    x_start * (1 - img_mask)
                    + init_img * img_mask
            )
        if self.task_type == 'inpainting':
            denoised_function = denoised_fn
        else:
            denoised_function = None
        self.model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 4, new_h, new_w),
            device=self.device,
            denoised_type=denoised_type,
            dynamic_threshold_v=dynamic_threshold_v,
            noise=noise,
            progress=progress,
            model_kwargs=model_kwargs,
            init_step=init_step,
            denoised_fn=denoised_function,
        )[:batch_size]
        self.model.del_cache()
        if self.use_image_enc:
            with torch.no_grad():
                samples = self.image_encoder.decode(samples / self.scale)
        samples = samples[:, :, :h, :w]
        return process_images(samples)

    def generate_text2img(self, prompt, num_steps=100,
                          batch_size=1, guidance_scale=7, progress=True,
                          dynamic_threshold_v=99.5, denoised_type='dynamic_threshold', h=512, w=512):
        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        return self.generate_img(prompt=prompt, batch_size=batch_size,
                                 diffusion=diffusion,
                                 guidance_scale=guidance_scale, progress=progress,
                                 dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                                    h=h, w=w)

    def generate_img2img(self, prompt, pil_img, strength=0.7,
                          num_steps=100, guidance_scale=7, progress=True,
                          dynamic_threshold_v=99.5, denoised_type='dynamic_threshold'):

        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        image = prepare_image(pil_img).to(self.device)
        image = self.image_encoder.encode(image).sample() * self.scale
        start_step = int(diffusion.num_timesteps * (1 - strength))
        image = q_sample(image, torch.tensor(diffusion.timestep_map[start_step - 1]).to(self.device),
                         schedule_name=config['diffusion_config']['noise_schedule'], num_steps=config['diffusion_config']['steps'])
        image = image.repeat(2, 1, 1, 1)
        return self.generate_img(prompt=prompt, batch_size=1,
                                 diffusion=diffusion, noise=image,
                                 guidance_scale=guidance_scale, progress=progress,
                                 dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                                 init_step=start_step)

    def generate_inpainting(self, prompt, pil_img, img_mask,
                          num_steps=100, guidance_scale=7, progress=True,
                          dynamic_threshold_v=99.5, denoised_type='dynamic_threshold'):
        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        image = prepare_image(pil_img).to(self.device)
        image = self.image_encoder.encode(image).sample() * self.scale
        image_shape = tuple(image.shape[-2:])
        img_mask = torch.from_numpy(img_mask).unsqueeze(0).unsqueeze(0)
        img_mask = F.interpolate(
            img_mask, image_shape, mode="nearest",
        ).to(self.device)
        image = image.repeat(2, 1, 1, 1)
        img_mask = img_mask.repeat(2, 1, 1, 1)
        return self.generate_img(prompt=prompt, batch_size=1,
                                 diffusion=diffusion,
                                 guidance_scale=guidance_scale, progress=progress,
                                 dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                                 init_img=image, img_mask=img_mask, )