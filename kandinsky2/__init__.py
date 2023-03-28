import os
from .kandinsky2_model import Kandinsky2
from .kandinsky2_1_model import Kandinsky2_1
from huggingface_hub import hf_hub_url, cached_download
from copy import deepcopy
from omegaconf.dictconfig import DictConfig

CONFIG_2_0 = {'model_config':
                  {'image_size': 64, 'num_channels': 384, 'num_res_blocks': 3, 'channel_mult': '', 'num_heads': 1,
                   'num_head_channels': 64, 'num_heads_upsample': -1, 'attention_resolutions': '32,16,8', 'dropout': 0,
                   'model_dim': 768, 'use_scale_shift_norm': True, 'resblock_updown': True, 'use_fp16': False,
                   'cache_text_emb': True, 'text_encoder_in_dim1': 1024, 'text_encoder_in_dim2': 640,
                   'pooling_type': 'from_model', 'in_channels': 4, 'out_channels': 8, 'up': False, 'inpainting': False},

              'diffusion_config': {'learn_sigma': True, 'sigma_small': False, 'steps': 1000, 'noise_schedule': 'linear',
                                   'timestep_respacing': '', 'use_kl': False, 'predict_xstart': False,
                                   'rescale_timesteps': True, 'rescale_learned_sigmas': True,
                                   'linear_start': 0.0001, 'linear_end': 0.02},
              'image_enc_params': {'name': 'AutoencoderKL', 'scale': 0.0512,
                                   'params': {'ckpt_path': '', 'embed_dim': 4,
                                              'ddconfig': {'double_z': True, 'z_channels': 4, 'resolution': 256,
                                                           'in_channels': 3, 'out_ch': 3, 'ch': 128,
                                                           'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2,
                                                           'attn_resolutions': [], 'dropout': 0.0}}},
              'text_enc_params1': {'model_path': '', 'model_name': 'multiclip'},
              'text_enc_params2': {'model_path': '', 'model_name': 'MT5EncoderModel'},
              'tokenizer_name1': '',
              'tokenizer_name2': ''}

CONFIG_2_1 = {'clip_name': 'ViT-L/14',
              'clip_image_size': 224,
              'tokenizer_name': '',
              'image_enc_params': {'name': 'MOVQ', 'scale': 1, 'ckpt_path': '',
                                   'params': {'embed_dim': 4, 'n_embed': 16384,
                                              'ddconfig': {'double_z': False, 'z_channels': 4, 'resolution': 256,
                                                           'in_channels': 3, 'out_ch': 3, 'ch': 128,
                                                           'ch_mult': [1, 2, 2, 4], 'num_res_blocks': 2,
                                                           'attn_resolutions': [32], 'dropout': 0.0}}},
              'text_enc_params': {'model_path': '', 'model_name': 'multiclip', 'in_features': 1024,
                                  'out_features': 768},
              'prior': {'clip_mean_std_path': 'cene655/mean_std/ViT-L-14_stats.th', 'params': {
                  'model': {'type': 'prior', 'diffusion_sampler': 'uniform',
                            'hparams': {'text_ctx': 77, 'xf_width': 2048, 'xf_layers': 20, 'xf_heads': 32,
                                        'xf_final_ln': True, 'xf_padding': False, 'text_drop': 0.2, 'clip_dim': 768,
                                        'clip_xf_width': 768}},
                  'diffusion': {'steps': 1000, 'learn_sigma': False, 'sigma_small': True, 'noise_schedule': 'cosine',
                                'use_kl': False, 'predict_xstart': True, 'rescale_learned_sigmas': False,
                                'timestep_respacing': ''}}},
              'model_config': {'version': '2.1', 'image_size': 64, 'num_channels': 384, 'num_res_blocks': 3,
                               'channel_mult': '',
                               'num_heads': 1, 'num_head_channels': 64, 'num_heads_upsample': -1,
                               'attention_resolutions': '32,16,8',
                               'dropout': 0, 'model_dim': 768, 'use_scale_shift_norm': True, 'resblock_updown': True,
                               'use_fp16': True,
                               'cache_text_emb': True, 'text_encoder_in_dim1': 1024, 'text_encoder_in_dim2': 768,
                               'image_encoder_in_dim': 768,
                               'num_image_embs': 10, 'pooling_type': 'from_model', 'in_channels': 4, 'out_channels': 8,
                               'use_flash_attention': False},
              'diffusion_config': {'learn_sigma': True, 'sigma_small': False, 'steps': 1000, 'noise_schedule': 'linear',
                                   'timestep_respacing': '', 'use_kl': False, 'predict_xstart': False,
                                   'rescale_timesteps': True, 'rescale_learned_sigmas': True, 'linear_start': 0.00085,
                                   'linear_end': 0.012}}


def get_kandinsky2(device, task_type='text2img', cache_dir='/tmp/kandinsky2', use_auth_token=None, model_version='2.1',
                   use_flash_attention=False):
    if model_version == '2.0':
        cache_dir = os.path.join(cache_dir, '2_0')
        config = deepcopy(CONFIG_2_0)
        if task_type == 'inpainting':
            model_name = 'Kandinsky-2-0-inpainting.pt'
            config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename=model_name)
        elif task_type == 'text2img':
            model_name = 'Kandinsky-2-0.pt'
            config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename=model_name)
        else:
            raise ValueError('Only text2img, img2img and inpainting is available')

        cached_download(config_file_url, cache_dir=cache_dir, force_filename=model_name,
                        use_auth_token=use_auth_token)

        cache_dir_text_en1 = os.path.join(cache_dir, 'text_encoder1')
        cache_dir_text_en2 = os.path.join(cache_dir, 'text_encoder2')
        for name in ['config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model', 'special_tokens_map.json',
                     'tokenizer.json', 'tokenizer_config.json']:
            config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename=f'text_encoder1/{name}')
            cached_download(config_file_url, cache_dir=cache_dir_text_en1, force_filename=name,
                            use_auth_token=use_auth_token)

        for name in ['config.json', 'pytorch_model.bin', 'spiece.model', 'special_tokens_map.json',
                     'tokenizer_config.json']:
            config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename=f'text_encoder2/{name}')
            cached_download(config_file_url, cache_dir=cache_dir_text_en2, force_filename=name,
                            use_auth_token=use_auth_token)
        config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename='vae.ckpt')
        cached_download(config_file_url, cache_dir=cache_dir, force_filename='vae.ckpt',
                        use_auth_token=use_auth_token)

        config['text_enc_params1']['model_path'] = cache_dir_text_en1
        config['text_enc_params2']['model_path'] = cache_dir_text_en2
        config['tokenizer_name1'] = cache_dir_text_en1
        config['tokenizer_name2'] = cache_dir_text_en2
        config['image_enc_params']['params']['ckpt_path'] = os.path.join(cache_dir, 'vae.ckpt')
        unet_path = os.path.join(cache_dir, model_name)

        model = Kandinsky2(config, unet_path, device, task_type)
    elif model_version == '2.1':
        cache_dir = os.path.join(cache_dir, '2_1')
        config = DictConfig(deepcopy(CONFIG_2_1))
        config['model_config']['use_flash_attention'] = use_flash_attention
        if task_type == 'text2img':
            model_name = 'decoder_fp16.ckpt'  # RENAME
            config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.1', filename=model_name)
        elif task_type == 'inpainting':
            model_name = 'inpainting_fp16.ckpt'  # RENAME
            config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.1', filename=model_name)
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=model_name,
                        use_auth_token=use_auth_token)
        prior_name = 'prior_fp16.ckpt'
        config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.1', filename=prior_name)
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=prior_name,
                        use_auth_token=use_auth_token)

        cache_dir_text_en = os.path.join(cache_dir, 'text_encoder')
        for name in ['config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model', 'special_tokens_map.json',
                     'tokenizer.json', 'tokenizer_config.json']:
            config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.1', filename=f'text_encoder/{name}')
            cached_download(config_file_url, cache_dir=cache_dir_text_en, force_filename=name,
                            use_auth_token=use_auth_token)

        config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.1', filename='movq_final.ckpt')
        cached_download(config_file_url, cache_dir=cache_dir, force_filename='movq_final.ckpt',
                        use_auth_token=use_auth_token)

        config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.1', filename='ViT-L-14_stats.th')
        cached_download(config_file_url, cache_dir=cache_dir, force_filename='ViT-L-14_stats.th',
                        use_auth_token=use_auth_token)

        config['tokenizer_name'] = cache_dir_text_en
        config['text_enc_params']['model_path'] = cache_dir_text_en
        config['prior']['clip_mean_std_path'] = os.path.join(cache_dir, 'ViT-L-14_stats.th')
        config['image_enc_params']['ckpt_path'] = os.path.join(cache_dir, 'movq_final.ckpt')
        cache_model_name = os.path.join(cache_dir, model_name)
        cache_prior_name = os.path.join(cache_dir, prior_name)
        model = Kandinsky2_1(config, cache_model_name, cache_prior_name, device, task_type=task_type)
    else:
        raise ValueError('Only 2.0 and 2.1 is available')
    return model
