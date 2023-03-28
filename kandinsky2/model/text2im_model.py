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
        text_encoder_in_dim1=1024,
        text_encoder_in_dim2=640,
        pooling_type="attention_pooling",  # ['from_model', 'attention_pooling']
        *args,
        cache_text_emb=True,
        **kwargs,
    ):
        self.model_dim = model_dim
        super().__init__(*args, **kwargs, encoder_channels=model_dim)
        self.pooling_type = pooling_type

        self.to_model_dim = nn.Linear(text_encoder_in_dim1, model_dim)

        if self.pooling_type == "from_model":
            self.proj = nn.Linear(text_encoder_in_dim2, self.model_channels * 4)
        elif self.pooling_type == "attention_pooling":
            self.proj = AttentionPooling(
                8, text_encoder_in_dim2, self.model_channels * 4
            )
        self.proj2 = AttentionPooling(8, 512, self.model_channels * 4)
        self.to_model_dim2 = nn.Linear(512, model_dim)
        self.ln_model1 = nn.LayerNorm(model_dim)
        self.ln_model2 = nn.LayerNorm(self.model_channels * 4)
        self.ln_model3 = nn.LayerNorm(self.model_channels * 4)
        self.cache_text_emb = cache_text_emb
        self.cache = None

    def convert_to_fp16(self):
        super().convert_to_fp16()
        self.proj.to(torch.float16)
        self.to_model_dim.to(torch.float16)
        self.to_model_dim2.to(torch.float16)
        self.proj2.to(torch.float16)
        self.ln_model1.to(torch.float16)
        self.ln_model2.to(torch.float16)
        self.ln_model3.to(torch.float16)

    def get_text_emb(
        self, full_emb1=None, pooled_emb1=None, full_emb2=None, pooled_emb2=None
    ):
        if self.cache is not None and self.cache_text_emb:
            return self.cache
        if self.pooling_type == "from_model":
            xf_proj = self.proj(pooled_emb1)
        elif self.pooling_type == "attention_pooling":
            xf_proj = self.proj(full_emb1)
        xf_proj = self.ln_model2(xf_proj)
        pooled_emb2 = self.ln_model3(self.proj2(full_emb2))
        xf_proj += pooled_emb2
        xf_out = self.ln_model1(
            torch.cat(
                [self.to_model_dim(full_emb1), self.to_model_dim2(full_emb2)], dim=1
            )
        )

        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL
        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = outputs
        return outputs

    def del_cache(self):
        self.cache = None

    def forward(
        self,
        x,
        timesteps,
        full_emb1=None,
        pooled_emb1=None,
        full_emb2=None,
        pooled_emb2=None,
    ):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        text_outputs = self.get_text_emb(
            full_emb1=full_emb1,
            pooled_emb1=pooled_emb1,
            full_emb2=full_emb2,
            pooled_emb2=pooled_emb2,
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
        h = h.type(torch.float32)
        h = self.out(h)
        return h


class InpaintText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
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
