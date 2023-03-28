import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import (
    T5EncoderModel,
    MT5EncoderModel,
    BertModel,
    XLMRobertaModel,
    AutoConfig,
    XLMRobertaModel,
)
import transformers
import os


def attention(q, k, v, d_k):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)
    return output


class AttentionPooling(nn.Module):
    def __init__(
        self,
        heads,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.d_k = in_dim // heads
        self.h = heads

        self.q_linear = nn.Linear(in_dim, in_dim)
        self.v_linear = nn.Linear(in_dim, in_dim)
        self.k_linear = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        bs = x.size(0)

        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.in_dim)

        output = self.out(concat)

        return output[:, 0]


class ImagenCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        transformer_width = 768
        embed_dim = 768
        transformer_layers = 12
        transformer_heads = transformer_width // 64
        vocab_size = 49408
        self.context_length = 77
        self.transformer = clip.model.Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = clip.model.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.out_proj.weight.dtype

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, mask=None):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        pooled_out = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        )
        return x, pooled_out


class MultilingualCLIP(nn.Module):
    def __init__(self, config, in_features=1024, out_features=640):
        super().__init__()
        loaded_config = AutoConfig.from_pretrained(config)
        self.transformer = XLMRobertaModel(loaded_config)
        self.LinearTransformation = torch.nn.Linear(
            in_features=in_features, out_features=out_features
        )

    def forward(self, input_ids, attention_mask):
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs2 = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
            dim=1
        )[:, None]
        return self.LinearTransformation(embs2), embs


class TextEncoder(nn.Module):
    def __init__(self, model_path, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "clip":
            self.model = ImagenCLIP()
            self.model.load_state_dict(torch.load(model_path))
        elif self.model_name == "T5EncoderModel":
            self.model = T5EncoderModel.from_pretrained(model_path)
        elif self.model_name == "MT5EncoderModel":
            self.model = MT5EncoderModel.from_pretrained(model_path)
        elif self.model_name == "BertModel":
            self.model = BertModel.from_pretrained(model_path)
        elif self.model_name == "multiclip":
            self.model = MultilingualCLIP(model_path, **kwargs)
            self.model.load_state_dict(
                torch.load(os.path.join(model_path, "pytorch_model.bin")), strict=False
            )
        elif self.model_name == "xlm_roberta":
            self.model = XLMRobertaModel.from_pretrained(model_path).half()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, tokens, mask=None):
        if self.model_name == "clip":
            full_out, pooled_out = self.model(tokens)
        elif self.model_name in ["T5EncoderModel", "MT5EncoderModel"]:
            pooled_out = None
            full_out = self.model(input_ids=tokens, attention_mask=mask)[
                "last_hidden_state"
            ]
        elif self.model_name in ["BertModel"]:
            out = self.model(input_ids=tokens, attention_mask=mask)
            full_out, pooled_out = out["last_hidden_state"], out["pooler_output"]
        elif self.model_name == "multiclip":
            pooled_out, full_out = self.model(input_ids=tokens, attention_mask=mask)
        elif self.model_name == "xlm_roberta":
            pooled_out = None
            full_out = self.model(input_ids=tokens, attention_mask=mask)[
                "last_hidden_state"
            ].float()
        return full_out, pooled_out
