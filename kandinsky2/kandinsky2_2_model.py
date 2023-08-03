from PIL import Image
import cv2
import torch
import math
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline, KandinskyV22InpaintPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel


class Kandinsky2_2:
    
    def __init__(
        self, 
        device, 
        task_type="text2img"
    ):
        self.device = device
        self.task_type = task_type
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained('kandinsky-community/kandinsky-2-2-prior', subfolder='image_encoder').to(torch.float16).to(self.device)
        if task_type == "text2img":
            self.unet = UNet2DConditionModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', subfolder='unet').to(torch.float16).to(self.device)
            self.prior = KandinskyV22PriorPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-prior', image_encoder=self.image_encoder, torch_dtype=torch.float16)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22Pipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', unet=self.unet, torch_dtype=torch.float16)
            self.decoder = self.decoder.to(self.device)
        elif task_type == "inpainting":
            self.unet = UNet2DConditionModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder-inpaint', subfolder='unet').to(torch.float16).to(self.device)
            self.prior = KandinskyV22PriorPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-prior', image_encoder=self.image_encoder, torch_dtype=torch.float16)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22InpaintPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder-inpaint', unet=self.unet, torch_dtype=torch.float16)
            self.decoder = self.decoder.to(self.device)
        elif task_type == "img2img":
            self.unet = UNet2DConditionModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', subfolder='unet').to(torch.float16).to(self.device)
            self.prior = KandinskyV22PriorPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-prior', image_encoder=self.image_encoder, torch_dtype=torch.float16)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22Img2ImgPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', unet=self.unet, torch_dtype=torch.float16)
            self.decoder = self.decoder.to(self.device)
        else:
            raise ValueError("Only text2img, img2img, inpainting is available")
            
    def get_new_h_w(self, h, w):
        new_h = h // 64
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        return new_h * 64, new_w * 64
    
    def generate_text2img(
        self,
        prompt,
        batch_size=1,
        decoder_steps=50,
        prior_steps=25,
        decoder_guidance_scale=4,
        prior_guidance_scale=4,
        h=512,
        w=512,
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        h, w = self.get_new_h_w(h, w)
        img_emb = self.prior(prompt=prompt, num_inference_steps=prior_steps,
                        num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale,
                        negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_decoder_prompt, num_inference_steps=prior_steps,
                             num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=decoder_steps, height=h,
                         width=w, guidance_scale=decoder_guidance_scale).images
        return images

    def generate_img2img(
        self,
        prompt,
        image,
        strength=0.4,
        batch_size=1,
        decoder_steps=100,
        prior_steps=25,
        decoder_guidance_scale=4,
        prior_guidance_scale=4,
        h=512,
        w=512,
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        h, w = self.get_new_h_w(h, w)
        img_emb = self.prior(prompt=prompt, num_inference_steps=prior_steps,
                        num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale,
                        negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=prior_steps,
                             num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=decoder_steps, height=h,
                         width=w, guidance_scale=decoder_guidance_scale,
                             strength=strength, image=image).images
        return images
    
    def mix_images(
        self,
        images_texts,
        weights,
        batch_size=1,
        decoder_steps=50,
        prior_steps=25,
        decoder_guidance_scale=4,
        prior_guidance_scale=4,
        h=512,
        w=512,
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        assert len(images_texts) == len(weights) and len(images_texts) > 0
        
        img_emb = self.prior.interpolate(images_and_prompts=images_texts, weights=weights,
                                         num_inference_steps=prior_steps, num_images_per_prompt=batch_size,
                                         guidance_scale=prior_guidance_scale, negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=prior_steps,
                             num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=decoder_steps, height=h,
                         width=w, guidance_scale=decoder_guidance_scale).images
        return images

    def generate_inpainting(
        self,
        prompt,
        pil_img,
        img_mask,
        batch_size=1,
        decoder_steps=50,
        prior_steps=25,
        decoder_guidance_scale=4,
        prior_guidance_scale=4,
        h=512,
        w=512,
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        
        img_emb = self.prior(prompt=prompt, num_inference_steps=prior_steps,
                        num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale,
                        negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=prior_steps,
                             num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=decoder_steps, height=h,
                         width=w, guidance_scale=decoder_guidance_scale,
                         image=pil_img, mask_image=img_mask).images
        return images
