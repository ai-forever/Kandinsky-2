import os
from .kandinsky2_model import Kandinsky2
from huggingface_hub import hf_hub_url, cached_download
from copy import deepcopy

CONFIG = {'model_config':
              {'image_size': 64, 'num_channels': 384, 'num_res_blocks': 3, 'channel_mult': '', 'num_heads': 1,
               'num_head_channels': 64, 'num_heads_upsample': -1, 'attention_resolutions': '32,16,8', 'dropout': 0,
               'model_dim': 768, 'use_scale_shift_norm': True, 'resblock_updown': True, 'use_fp16': True,
               'cache_text_emb': True, 'text_encoder_in_dim1': 1024, 'text_encoder_in_dim2': 640,
               'pooling_type': 'from_model', 'in_channels': 4, 'out_channels': 8, 'up': False, 'inpainting': False},

          'diffusion_config': {'learn_sigma': True, 'sigma_small': False, 'steps': 1000, 'noise_schedule': 'linear',
                               'timestep_respacing': '', 'use_kl': False, 'predict_xstart': False,
                               'rescale_timesteps': True, 'rescale_learned_sigmas': True},
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

def get_kandinsky2(device, task_type='text2img', cache_dir='/tmp/kandinsky2', use_auth_token=None):
    config = deepcopy(CONFIG)
    if task_type == 'inpainting':
        model_name = 'Kandinsky-2-0-inpainting.pt'
        config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename=model_name)
    elif task_type == 'text2img' or task_type == 'img2img':
        model_name = 'Kandinsky-2-0.pt'
        config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename=model_name)
    else:
        raise ValueError('Only text2img, img2img and inpainting is available')
        
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=model_name,
                        use_auth_token=use_auth_token)
    
    cache_dir_text_en1 = os.path.join(cache_dir, 'text_encoder1')
    cache_dir_text_en2 = os.path.join(cache_dir, 'text_encoder2')
    for name in ['config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']:  
        config_file_url = hf_hub_url(repo_id='sberbank-ai/Kandinsky_2.0', filename=f'text_encoder1/{name}')
        cached_download(config_file_url, cache_dir=cache_dir_text_en1, force_filename=name,
                        use_auth_token=use_auth_token)
        
    for name in ['config.json', 'pytorch_model.bin', 'spiece.model', 'special_tokens_map.json', 'tokenizer_config.json']:  
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
    return model
