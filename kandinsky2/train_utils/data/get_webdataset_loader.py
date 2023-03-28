import webdataset as wds
import webdataset
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import re
import boto3
import os
import botocore
from webdataset.handlers import reraise_exception
from webdataset.tariterators import tar_file_iterator
from webdataset import pipelinefilter

import torch
import sys, time
from dataclasses import dataclass, field
from itertools import islice
from typing import List
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import fsspec
from copy import deepcopy

from webdataset.filters import pipelinefilter
from webdataset.pytorch import IterableDataset
from webdataset import utils
import clip
from transformers import AutoTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm


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


def filter0(s):
    start = 0
    for i in range(len(s)):
        if s[i] != "0":
            start = i
            break
    return s[start:]


def make_seed(*args):
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


def create_path(item, m_dir):
    splited_dir = item["__url__"].split("/")
    new_path = (
        os.path.join(m_dir, splited_dir[-2])
        + "/"
        + splited_dir[-1][:-4]
        + "/"
        + item["fname"][:-4]
        + ".npy"
    )
    return new_path


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = (
            utils.pytorch_worker_seed if worker_seed is None else worker_seed
        )
        self.deterministic = deterministic
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if self.deterministic:
            seed = make_seed(self.worker_seed(), self.epoch)
        else:
            seed = make_seed(
                self.worker_seed(),
                self.epoch,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")
        self.rng = random.Random(seed)
        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.urls) - 1)
            yield dict(url=self.urls[index])


class NewSimpleShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        storage_options=None,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.storage_options = storage_options

    def __iter__(self):
        for ind in range(len(self.urls)):
            # print('ind=',ind)
            yield dict(url=self.urls[ind], storage_options=self.storage_options)


def init_webdataset(storage_options, drop_duplicates=True):
    def get_bytes_io(path):
        with fsspec.open(
            path, s3=storage_options, mode="rb", skip_instance_cache=True
        ) as f:
            byte_io = io.BytesIO(f.read())
        byte_io.seek(0)
        return byte_io

    def tar2csv(texts):
        for text in texts.split():
            if text[-4:] == ".tar":
                return text[:-3] + "csv"

    def url_opener(data, handler=reraise_exception, **kw):
        for sample in data:
            url = sample["url"]
            _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", url)
            try:
                stream = get_bytes_io(url)
                sample.update(stream=stream)
                yield sample
            except Exception as exn:
                exn.args = exn.args + (url,)
                if handler(exn):
                    continue
                else:
                    break

    def tar_file_expander(data, handler=reraise_exception):
        """Expand a stream of open tar files into a stream of tar file contents.

        This returns an iterator over (filename, file_contents).
        """
        for source in data:
            url = source["url"]
            try:
                if storage_options is None:
                    df = pd.read_csv(tar2csv(url))
                else:
                    df = pd.read_csv(tar2csv(url), storage_options=storage_options)

                if drop_duplicates:
                    df = df[~df["image_name"].duplicated()]
                df.set_index("image_name", inplace=True)
                df = df.to_dict("index")

                assert isinstance(source, dict)
                assert "stream" in source

                for sample in tar_file_iterator(source["stream"]):
                    assert (
                        isinstance(sample, dict)
                        and "data" in sample
                        and "fname" in sample
                    )
                    sample["__url__"] = url
                    df_row = df[sample["fname"]]
                    sample.update(df_row)
                    yield sample
            except Exception as exn:
                exn.args = exn.args + (source.get("stream"), source.get("url"))
                if handler(exn):
                    continue
                else:
                    break
            else:
                del df

    def tarfile_samples(src, handler=reraise_exception):
        streams = url_opener(src, handler=handler)
        files = tar_file_expander(streams, handler=handler)
        return files

    webdataset.tariterators.url_opener = url_opener
    webdataset.tariterators.tar_file_expander = tar_file_expander
    webdataset.tariterators.tarfile_to_samples = pipelinefilter(tarfile_samples)


def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def create_webdataset(
    urls_path,
    storage_options,
    resolution=256,
    tokenizer_name=None,
    is_val=False,
    drop_emb_prob=0.1,
    seq_len=77,
    emb_folder=None,
    is_up=False,
    low_res=64,
    caption_columns=None,
):
    init_webdataset(storage_options)
    urls = pd.read_csv(urls_path)["urls"].values
    if is_val:
        dataset = wds.WebDataset(NewSimpleShards(urls), handler=wds.warn_and_continue)
    else:
        dataset = wds.WebDataset(ResampledShards(urls), handler=wds.warn_and_continue)
    if tokenizer_name != "clip" and emb_folder is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    transform1 = _transform(224)

    def tokenize(text):
        out_dict = {}
        if np.random.binomial(1, 0.5):
            text = ""
        if tokenizer_name != "clip":
            text_encoding = tokenizer(
                text,
                max_length=seq_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            out_dict["tokens"] = text_encoding["input_ids"][0]
            out_dict["mask"] = text_encoding["attention_mask"][0]
        else:
            out_dict["tokens"] = clip.tokenize(text, truncate=True)[0]
        return out_dict

    def filter_dataset(item):
        if "pwatermark" in item and item["pwatermark"] > 0.25:
            return False
        try:
            image_data = item["data"]

            pil_image_original = Image.open(io.BytesIO(image_data))
            pil_image_original.load()

            pil_image = np.array(pil_image_original.convert("RGB"))

            if pil_image.shape[0] < 600 or pil_image.shape[1] < 600:
                return False
            return True
        except:
            return False

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        image_data = item["data"]
        if emb_folder is None:
            for caption in caption_columns:
                if caption in item:
                    if type(item[caption]) == type(""):
                        text = item[caption]  # item['eng_caption']
            text = tokenize(text)
        else:
            emb_path = create_path(item, emb_folder)
            text = {"full_emb": torch.from_numpy(np.load(emb_path)).float()}

        pil_image_original = Image.open(io.BytesIO(image_data))
        pil_image_original.load()
        clip_image = transform1(deepcopy(pil_image_original))
        if np.random.binomial(1, 0.1):
            text["clip_image"] = torch.zeros(3, 224, 224)
        else:
            text["clip_image"] = clip_image
        pil_image_original = center_crop(pil_image_original)
        pil_image = pil_image_original.resize(
            (768, 768), resample=Image.BICUBIC, reducing_gap=1
        )
        arr = np.array(pil_image.convert("RGB"))
        arr = arr.astype(np.float32) / 127.5 - 1
        if is_up:
            pil_image_low = pil_image_original.resize(
                (low_res, low_res), resample=Image.BICUBIC, reducing_gap=1
            )
            pil_image_low = np.array(pil_image_low.convert("RGB"))
            pil_image_low = pil_image_low.astype(np.float32) / 127.5 - 1
            text["low_res"] = np.transpose(pil_image_low, [2, 0, 1])
            return np.transpose(arr, [2, 0, 1]), text
        else:
            return np.transpose(arr, [2, 0, 1]), text

    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.warn_and_continue
    )
    return transformed_dataset


def create_webdataset_loader(
    urls_path,
    storage_options,
    resolution=256,
    tokenizer_name=None,
    is_val=False,
    drop_emb_prob=0.1,
    seq_len=77,
    emb_folder=None,
    is_up=False,
    low_res=64,
    caption_columns=None,
    batch_size=8,
    num_workers=8,
):
    dataset = create_webdataset(
        urls_path=urls_path,
        storage_options=storage_options,
        resolution=resolution,
        tokenizer_name=tokenizer_name,
        is_val=is_val,
        drop_emb_prob=drop_emb_prob,
        seq_len=seq_len,
        emb_folder=emb_folder,
        is_up=is_up,
        low_res=low_res,
        caption_columns=caption_columns,
    )
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )


class LightningDataModule(pl.LightningDataModule):
    """PyTorch Lightning data class"""

    def __init__(self, train_data_config, val_data_config):
        super().__init__()
        self.train_data_config = train_data_config
        self.val_data_config = val_data_config

    def train_dataloader(self):
        return create_webdataset_loader(**self.train_data_config)

    def test_dataloader(self):
        return create_webdataset_loader(**self.val_data_config)

    def val_dataloader(self):
        return create_webdataset_loader(**self.val_data_config)
