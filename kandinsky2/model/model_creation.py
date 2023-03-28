from .gaussian_diffusion import get_named_beta_schedule
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .text2im_model import Text2ImUNet, InpaintText2ImUNet
from .text2im_model2_1 import Text2ImUNet as Text2ImUNet2_1
from .text2im_model2_1 import InpaintText2ImUNet as InpaintText2ImUNet2_1


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    model_dim,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    text_encoder_in_dim1,
    text_encoder_in_dim2,
    pooling_type,
    in_channels,
    out_channels,
    up,
    inpainting,
    version="2.0",
    **kwargs,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    if inpainting:
        if version == "2.0":
            model_cls = InpaintText2ImUNet
        elif version == "2.1":
            model_cls = InpaintText2ImUNet2_1
        else:
            ValueError("Only 2.0 and 2.1 versions are available")
    else:
        if version == "2.0":
            model_cls = Text2ImUNet
        elif version == "2.1":
            model_cls = Text2ImUNet2_1
        else:
            ValueError("Only 2.0 and 2.1 versions are available")
    return model_cls(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        model_dim=model_dim,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
        text_encoder_in_dim1=text_encoder_in_dim1,
        text_encoder_in_dim2=text_encoder_in_dim2,
        pooling_type=pooling_type,
        **kwargs,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    linear_start=0.0001,
    linear_end=0.02,
):
    betas = get_named_beta_schedule(
        noise_schedule, steps, linear_start=linear_start, linear_end=linear_end
    )
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
