import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from .nn import timestep_embedding
from .unet import UNetModel
import math
from abc import abstractmethod
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .text_encoders import AttentionPooling


class Text2ImUNet(UNetModel):
    def __init__(
        self,
        model_dim,
        image_encoder_in_dim=768,
        text_encoder_in_dim1=1024,
        text_encoder_in_dim2=768,
        num_image_embs=10,
        pooling_type="attention_pooling",  # ['from_model', 'attention_pooling']
        *args,
        cache_text_emb=True,
        **kwargs,
    ):
        self.model_dim = model_dim
        super().__init__(*args, **kwargs, encoder_channels=model_dim)
        self.pooling_type = pooling_type

        self.num_image_embs = num_image_embs
        self.clip_to_seq = nn.Linear(
            image_encoder_in_dim, model_dim * self.num_image_embs
        )

        self.to_model_dim_n = nn.Linear(text_encoder_in_dim1, model_dim)

        if self.pooling_type == "from_model":
            self.proj_n = nn.Linear(text_encoder_in_dim2, self.model_channels * 4)
        elif self.pooling_type == "attention_pooling":
            self.proj_n = AttentionPooling(
                8, text_encoder_in_dim1, self.model_channels * 4
            )
        self.ln_model_n = nn.LayerNorm(self.model_channels * 4)
        self.img_layer = nn.Linear(image_encoder_in_dim, self.model_channels * 4)
        self.cache_text_emb = cache_text_emb
        self.cache = None
        self.model_dim = model_dim

    def convert_to_fp16(self):
        super().convert_to_fp16()
        self.clip_to_seq.to(torch.float16)
        self.proj_n.to(torch.float16)
        self.to_model_dim_n.to(torch.float16)
        self.ln_model_n.to(torch.float16)
        self.img_layer.to(torch.float16)

    def get_text_emb(self, full_emb=None, pooled_emb=None, image_emb=None):
        if self.cache is not None and self.cache_text_emb:
            return self.cache

        clip_seq = self.clip_to_seq(image_emb).reshape(
            image_emb.shape[0], self.num_image_embs, self.model_dim
        )

        if self.pooling_type == "from_model":
            xf_proj = self.proj_n(pooled_emb)
        elif self.pooling_type == "attention_pooling":
            xf_proj = self.proj_n(full_emb)

        xf_proj = self.ln_model_n(xf_proj)
        if image_emb is not None:
            xf_proj = xf_proj + self.img_layer(image_emb)
        xf_out = torch.cat((clip_seq, self.to_model_dim_n(full_emb)), dim=1)

        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL
        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = outputs
        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, full_emb=None, pooled_emb=None, image_emb=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        text_outputs = self.get_text_emb(
            full_emb=full_emb, pooled_emb=pooled_emb, image_emb=image_emb
        )
        xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
        emb = emb + xf_proj.to(emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h


class SuperResText2ImUNet(Text2ImUNet):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        x = torch.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class InpaintText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = torch.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = torch.zeros_like(x[:, :1])
        return super().forward(
            torch.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )
