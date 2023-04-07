import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
import re
import os

import torch
import sys, time
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from copy import deepcopy

from transformers import AutoTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from random import randint
from ...model.prior import CustomizedTokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class TextImageDataset(Dataset):
    def __init__(
        self,
        df_path,
        clip_image_size=224,
        drop_text_prob=0.1,
        infinity=False,
    ):
        self.df = pd.read_csv(df_path)
        self.tokenizer = CustomizedTokenizer()
        self.transform1 = _transform(clip_image_size)
        self.drop_text_prob = drop_text_prob
        self.clip_image_size = clip_image_size
        self.infinity = infinity

    def __len__(self):
        if self.infinity:
            return 99999999
        else:
            return len(self.df)

    def __getitem__(self, item):
        if self.infinity:
            ind = randint(0, len(self.df) - 1)
        else:
            ind = item
        out_dict = {}
        image = Image.open(self.df["image_name"].iloc[ind])
        clip_image = self.transform1(image)
        if np.random.binomial(1, self.drop_text_prob):
            text = ""
        else:
            text = self.df["caption"].iloc[ind]
        out_dict["tokens"], out_dict["mask"] = self.tokenizer.padded_tokens_and_mask([text], 77)
        out_dict["tokens"] = out_dict["tokens"][0]
        out_dict["mask"] = out_dict["mask"][0]
        return clip_image, out_dict


def create_loader(batch_size, num_workers, shuffle=False, **dataset_params):
    dataset = TextImageDataset(**dataset_params)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


class LightningDataModule(pl.LightningDataModule):
    """PyTorch Lightning data class"""

    def __init__(self, train_config, val_config):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config

    def train_dataloader(self):
        return create_loader(**self.train_config)

    def test_dataloader(self):
        return create_loader(**self.val_config)

    def val_dataloader(self):
        return create_loader(**self.val_config)
