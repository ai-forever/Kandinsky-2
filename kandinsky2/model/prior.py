import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
from .model_creation import create_gaussian_diffusion

import clip

from clip.simple_tokenizer import SimpleTokenizer, default_bpe


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period)
        * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


'''
class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x.float()).to(x.dtype)
'''


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x).to(x.dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x, mask=None):
        x = self.c_qkv(x)
        x = self.attention(x, mask=mask)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv, mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        weight = th.einsum("bthc,bshc->bhts", q * scale, k * scale)
        wdtype = weight.dtype
        if mask is not None:
            weight = weight + mask[:, None, ...]
        weight = th.softmax(weight, dim=-1).type(wdtype)
        return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.resblocks:
            x = block(x, mask=mask)
        return x


class PriorTransformer(nn.Module):
    """
    A Causal Transformer that conditions on CLIP text embedding, text.
    Expects an extra kwarg `tokens` of text.
    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    """

    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        xf_padding,
        clip_dim,
        clip_xf_width,
    ):
        super().__init__()

        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_layers = xf_layers
        self.xf_heads = xf_heads
        self.xf_padding = xf_padding
        self.clip_dim = clip_dim
        self.clip_xf_width = clip_xf_width
        self.ext_len = 4

        self.time_embed = nn.Sequential(
            nn.Linear(xf_width, xf_width),
            nn.SiLU(),
            nn.Linear(xf_width, xf_width),
        )
        self.text_enc_proj = nn.Linear(clip_xf_width, xf_width)
        self.text_emb_proj = nn.Linear(clip_dim, xf_width)
        self.clip_img_proj = nn.Linear(clip_dim, xf_width)
        self.out_proj = nn.Linear(xf_width, clip_dim)
        self.transformer = Transformer(
            text_ctx + self.ext_len,
            xf_width,
            xf_layers,
            xf_heads,
        )
        if xf_final_ln:
            self.final_ln = LayerNorm(xf_width)
        else:
            self.final_ln = None

        self.positional_embedding = nn.Parameter(
            th.empty(1, text_ctx + self.ext_len, xf_width)
        )
        self.prd_emb = nn.Parameter(th.randn((1, 1, xf_width)))

        if self.xf_padding:
            self.padding_embedding = nn.Parameter(
                th.empty(text_ctx + self.ext_len, xf_width)
            )
            nn.init.normal_(self.padding_embedding, std=0.01)

        nn.init.normal_(self.prd_emb, std=0.01)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(
        self,
        x,
        timesteps,
        text_emb=None,
        text_enc=None,
        mask=None,
        causal_mask=None,
    ):
        x = x.to(self.text_enc_proj.weight.dtype)
        text_emb = text_emb.to(self.text_enc_proj.weight.dtype)
        text_enc = text_enc.to(self.text_enc_proj.weight.dtype)
        bsz = x.shape[0]
        mask = F.pad(mask, (0, self.ext_len), value=True)
        t_emb = self.time_embed(
            timestep_embedding(timesteps, self.xf_width).to(x.dtype)
        )
        text_enc = self.text_enc_proj(text_enc)
        text_emb = self.text_emb_proj(text_emb)
        x = self.clip_img_proj(x)

        input_seq = [
            text_enc,
            text_emb[:, None, :],
            t_emb[:, None, :],
            x[:, None, :],
            self.prd_emb.to(x.dtype).expand(bsz, -1, -1),
        ]
        input = th.cat(input_seq, dim=1)
        input = input + self.positional_embedding.to(input.dtype)
        if self.xf_padding:
            input = th.where(
                mask[..., None], input, self.padding_embedding[None].to(input.dtype)
            )

        mask = th.where(mask, 0.0, float("-inf"))
        mask = (mask[:, None, :] + causal_mask).float()

        out = self.transformer(input, mask=mask)
        if self.final_ln is not None:
            out = self.final_ln(out)

        out = self.out_proj(out[:, -1])

        return out


class PriorDiffusionModel(torch.nn.Module):
    def __init__(self, config, tokenizer, clip_mean, clip_std):
        super().__init__()

        self._conf = config
        self._model_conf = config.model.hparams
        self._diffusion_kwargs = dict(
            steps=config.diffusion.steps,
            learn_sigma=config.diffusion.learn_sigma,
            sigma_small=config.diffusion.sigma_small,
            noise_schedule=config.diffusion.noise_schedule,
            use_kl=config.diffusion.use_kl,
            predict_xstart=config.diffusion.predict_xstart,
            rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
            timestep_respacing=config.diffusion.timestep_respacing,
        )
        self._tokenizer = tokenizer

        self.register_buffer("clip_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("clip_std", clip_std[None, :], persistent=False)

        causal_mask = self.get_causal_mask()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.model = PriorTransformer(
            text_ctx=self._model_conf.text_ctx,
            xf_width=self._model_conf.xf_width,
            xf_layers=self._model_conf.xf_layers,
            xf_heads=self._model_conf.xf_heads,
            xf_final_ln=self._model_conf.xf_final_ln,
            xf_padding=self._model_conf.xf_padding,
            clip_dim=self._model_conf.clip_dim,
            clip_xf_width=self._model_conf.clip_xf_width,
        )

        cf_token, cf_mask = self.set_cf_text_tensor()
        self.register_buffer("cf_token", cf_token, persistent=False)
        self.register_buffer("cf_mask", cf_mask, persistent=False)

    def set_cf_text_tensor(self):
        return self._tokenizer.padded_tokens_and_mask([""], self.model.text_ctx)

    def create_prior_diffusion(self):
        return create_gaussian_diffusion(**self._diffusion_kwargs)

    def get_sample_fn(self, timestep_respacing):
        use_ddim = timestep_respacing.startswith(("ddim", "fast"))

        diffusion_kwargs = copy.deepcopy(self._diffusion_kwargs)
        diffusion_kwargs.update(timestep_respacing=timestep_respacing)
        diffusion = create_gaussian_diffusion(**diffusion_kwargs)
        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop

        return sample_fn

    def get_causal_mask(self):
        seq_len = self._model_conf.text_ctx + 4
        mask = torch.empty(seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        mask = mask[None, ...]
        return mask

    def forward(
        self,
        txt_feat,
        txt_feat_seq,
        mask,
        cf_guidance_scales=None,
        timestep_respacing=None,
        denoised_fn=True,
    ):
        # cfg should be enabled in inference
        assert cf_guidance_scales is not None and all(cf_guidance_scales > 0.0)

        bsz_ = txt_feat.shape[0]
        bsz = bsz_ // 2

        def guided_model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = (
                model_out[:, : int(x_t.shape[1])],
                model_out[:, int(x_t.shape[1]) :],
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cf_guidance_scales.view(-1, 1) * (
                cond_eps - uncond_eps
            )
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        cond = {
            "text_emb": txt_feat,
            "text_enc": txt_feat_seq,
            "mask": mask,
            "causal_mask": self.causal_mask,
        }
        sample_fn = self.get_sample_fn(timestep_respacing)
        sample = sample_fn(
            guided_model_fn,
            (bsz_, self.model.clip_dim),
            noise=None,
            device=txt_feat.device,
            clip_denoised=False,
            denoised_fn=lambda x: torch.clamp(x, -10, 10),
            model_kwargs=cond,
        )
        sample = (sample * self.clip_std) + self.clip_mean

        return sample[:bsz]


class CustomizedTokenizer(SimpleTokenizer):
    def __init__(self):
        super().__init__(bpe_path=default_bpe())

        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]

    def padded_tokens_and_mask(self, texts, text_ctx):
        assert isinstance(texts, list) and all(
            isinstance(elem, str) for elem in texts
        ), "texts should be a list of strings"

        all_tokens = [
            [self.sot_token] + self.encode(text) + [self.eot_token] for text in texts
        ]

        mask = [
            [True] * min(text_ctx, len(tokens))
            + [False] * max(text_ctx - len(tokens), 0)
            for tokens in all_tokens
        ]
        mask = torch.tensor(mask, dtype=torch.bool)
        result = torch.zeros(len(all_tokens), text_ctx, dtype=torch.int)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > text_ctx:
                tokens = tokens[:text_ctx]
                tokens[-1] = self.eot_token
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result, mask
