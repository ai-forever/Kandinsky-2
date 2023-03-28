import math
import torch
import torch as th
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import importlib
from matplotlib.path import Path


def get_polygon_mask_params(
    mask_size, box, num_vertices, mask_scale, min_scale, max_scale
):
    center = ((box[2] + box[0]) / 2, (box[3] + box[1]) / 2)
    sizes = (box[2] - box[0], box[3] - box[1])

    part_avg_radii = np.linspace(
        mask_scale * sizes[0] / 2, mask_scale * sizes[1] / 2, num_vertices // 4
    )
    part_avg_radii = np.clip(
        part_avg_radii, min_scale * min(mask_size), max_scale * min(mask_size)
    )
    avg_radii = np.concatenate(
        [
            part_avg_radii,
            part_avg_radii[::-1],
            part_avg_radii,
            part_avg_radii[::-1],
        ]
    )
    return center, avg_radii


def smooth_cerv(x, y):
    num_vertices = x.shape[0]
    x = np.concatenate((x[-3:-1], x, x[1:3]))
    y = np.concatenate((y[-3:-1], y, y[1:3]))
    t = np.arange(x.shape[0])

    ti = np.linspace(2, num_vertices + 1, 4 * num_vertices)
    xi = interp1d(t, x, kind="quadratic")(ti)
    yi = interp1d(t, y, kind="quadratic")(ti)
    return xi, yi


def get_polygon_mask(mask_size, mask_points):
    x, y = np.meshgrid(np.arange(mask_size[0]), np.arange(mask_size[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    path = Path(mask_points)
    grid = path.contains_points(points)
    grid = grid.reshape((mask_size[0], mask_size[1]))
    return 1.0 - grid.astype(np.int32)


def generate_polygon(
    mask_size, center, num_vertices, radii, radii_var, angle_var, smooth=True
):
    angle_steps = np.random.uniform(
        1.0 - angle_var, 1.0 + angle_var, size=(num_vertices,)
    )
    angle_steps = 2 * np.pi * angle_steps / angle_steps.sum()

    radii = np.random.normal(radii, radii_var * radii)
    radii = np.clip(radii, 0, 2 * radii)
    angles = np.cumsum(angle_steps)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    if smooth:
        x, y = smooth_cerv(x, y)
    points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=-1)
    points = list(map(tuple, points.tolist()))
    return get_polygon_mask(mask_size, points)


def generate_circle_frame(mask_size, side_scales, num_vertices, radii_var, smooth=True):
    num_vertices_per_side = num_vertices // 4
    x_size, y_size = mask_size
    up_radii = np.array([y_size * (1.0 - side_scales[0]) // 2] * num_vertices_per_side)
    down_radii = np.array(
        [y_size * (1.0 - side_scales[1]) // 2] * num_vertices_per_side
    )
    left_radii = np.array(
        [x_size * (1.0 - side_scales[2]) // 2] * num_vertices_per_side
    )
    right_radii = np.array(
        [x_size * (1.0 - side_scales[3]) // 2] * num_vertices_per_side
    )

    center = (x_size // 2, y_size // 2)
    radii = np.concatenate(
        [
            right_radii[num_vertices_per_side // 2 :],
            down_radii,
            left_radii,
            up_radii,
            right_radii[: num_vertices_per_side // 2],
        ]
    )
    return 1.0 - generate_polygon(
        mask_size, center, num_vertices, radii, radii_var, 0.0, smooth=smooth
    )


def generate_square_frame(mask_size, side_scales, num_vertices, radii_var, smooth=True):
    num_vertices_per_side = num_vertices // 4
    x_size, y_size = mask_size
    diag_size = np.sqrt(x_size**2 + y_size**2)

    up_radii = np.linspace(
        diag_size * (1.0 - side_scales[0]) // 2,
        y_size * (1.0 - side_scales[0]) // 2,
        num_vertices_per_side // 2,
    )
    down_radii = np.linspace(
        diag_size * (1.0 - side_scales[1]) // 2,
        y_size * (1.0 - side_scales[1]) // 2,
        num_vertices_per_side // 2,
    )
    left_radii = np.linspace(
        diag_size * (1.0 - side_scales[2]) // 2,
        x_size * (1.0 - side_scales[2]) // 2,
        num_vertices_per_side // 2,
    )
    right_radii = np.linspace(
        diag_size * (1.0 - side_scales[3]) // 2,
        x_size * (1.0 - side_scales[3]) // 2,
        num_vertices_per_side // 2,
    )

    center = (x_size // 2, y_size // 2)
    radii = np.concatenate(
        [
            right_radii[::-1],
            down_radii,
            down_radii[::-1],
            left_radii,
            left_radii[::-1],
            up_radii,
            up_radii[::-1],
            right_radii,
        ]
    )
    return 1.0 - generate_polygon(
        mask_size, center, num_vertices, radii, radii_var, 0.0, smooth=smooth
    )


def generate_mask(mask_size, box, box_prob=0.1):
    mask = np.ones(mask_size)
    if np.random.binomial(1, box_prob):
        box = [int(i) for i in box]
        mask[box[1] : box[3], box[0] : box[2]] = 0

    else:
        actions = np.random.randint(0, 2, (2,))
        if 0 in actions:
            num_vertices = 16
            center, radii = get_polygon_mask_params(
                mask_size,
                box,
                num_vertices,
                mask_scale=1.5,
                min_scale=0.1,
                max_scale=0.6,
            )
            mask *= generate_polygon(
                mask_size, center, num_vertices, radii, radii_var=0.15, angle_var=0.15
            )
        if 1 in actions:
            radii_var = 0.15 * np.random.random()
            num_vertices = np.random.choice([16, 32])
            if np.random.random() < 0.5:
                side_scales = 0.25 * np.random.random((4,)) + 0.05
                mask *= generate_square_frame(
                    mask_size, side_scales, num_vertices, radii_var=radii_var
                )
            else:
                side_scales = 0.15 * np.random.random((4,)) + 0.1
                mask *= generate_circle_frame(
                    mask_size, side_scales, num_vertices, radii_var=radii_var
                )
    return mask


def get_boxes(bs, target_size, min_scale=0.1, max_scale=0.62):
    min_size_x = min_scale * target_size[0]
    max_size_x = max_scale * target_size[0]
    min_size_y = min_scale * target_size[1]
    max_size_y = max_scale * target_size[1]

    boxes_size_x = (max_size_x - min_size_x) * np.random.random((bs, 1)) + min_size_x
    boxes_size_y = (max_size_y - min_size_y) * np.random.random((bs, 1)) + min_size_y

    x0 = (target_size[0] - max_size_x) * np.random.random((bs, 1))
    y0 = (target_size[1] - max_size_y) * np.random.random((bs, 1))

    boxes = np.concatenate((x0, y0, x0 + boxes_size_x, y0 + boxes_size_x), -1)
    return boxes.tolist()


def get_image_mask(bs, target_size):
    boxes = get_boxes(bs, target_size)
    image_mask = torch.stack(
        [torch.tensor(generate_mask(target_size, box)) for box in boxes]
    )
    return image_mask


def freeze_decoder(
    model,
    freeze_resblocks=False,
    freeze_attention=False,
):
    for name, p in model.named_parameters():
        name = name.lower()
        if (
            "in_layers" in name
            or "h_upd" in name
            or "x_upd" in name
            or "emb_layers" in name
            or "out_layers" in name
        ):
            p.requires_grad = not freeze_resblocks
        elif "proj_out" in name or "qkv" in name:
            p.requires_grad = not freeze_attention
    return model
