import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import importlib

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def dict_keys(d, keys):
    d2 = {}
    for i in keys:
        d2[i] = d[i]
    return d2


def return_images(bath):
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()


from PIL import Image


def prepare_image(pil_image):
    w, h = pil_image.size
    pil_image = pil_image.resize((512, 512), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def q_sample(x_start, t, schedule_name='linear', num_steps=1000, noise=None):
    betas = get_named_beta_schedule(schedule_name, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    if noise is None:
        noise = torch.randn_like(x_start)
    assert noise.shape == x_start.shape
    return (
            _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
    )


def process_images(batch):
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).to('cpu').permute(0, 2, 3, 1).numpy()
    images = []
    for i in range(scaled.shape[0]):
        images.append(Image.fromarray(scaled[i]))
    return images
