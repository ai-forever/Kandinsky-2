# Kandinsky 2.0

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface-yello.svg)]([https://huggingface.co/spaces/shi-labs/OneFormer](https://huggingface.co/sberbank-ai/Kandinsky_2.0))

`pip install "git+https://github.com/ai-forever/Kandinsky-2.0.git"`

## Model architecture:

It is a latent diffusion model with two multilingual text encoders:
* mCLIP-XLMR 560M parameters
* mT5-encoder-small 146M parameters

These encoders and multilingual training datasets unveil the real multilingual text2image generation experience!

Kandinsky 2.0 was trained on a large 1B multilingual set, including samples that we used to train Kandinsky.

In terms of diffusion architecture Kandinsky 2.0 implements UNet with 1.2B parameters.

Kandinsky 2.0 architecture overview:

![](./content/NatallE.png)

# How to use:
 
 Check our jupyter notebooks with examples in `./notebooks` folder
 
## 1. text2img

```python
from kandinsky2 import get_kandinsky2

model = get_kandinsky2('cuda', task_type='text2img')
images = model.generate_text2img('A teddy bear на красной площади', batch_size=4, h=512, w=512, num_steps=75, denoised_type='dynamic_threshold', dynamic_threshold_v=99.5, sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=10)
```
![](./content/bear.jpeg)

prompt: "A teddy bear на красной площади"

## 2. inpainting
```python 
from kandinsky2 import get_kandinsky2
from PIL import Image
import numpy as np

model = get_kandinsky2('cuda', task_type='inpainting')
init_image = Image.open('image.jpg')
mask = np.ones((512, 512), dtype=np.float32)
mask[100:] =  0
images = model.generate_inpainting('Девушка в красном платье', init_image, mask, num_steps=50, denoised_type='dynamic_threshold', dynamic_threshold_v=99.5, sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=10)
```

![](./content/inpainting.png)

prompt: "Девушка в красном платье"

## 3. img2img
```python
from kandinsky2 import get_kandinsky2
from PIL import Image

model = get_kandinsky2('cuda', task_type='img2img')
init_image = Image.open('image.jpg')
images = model.generate_img2img('кошка', init_image, strength=0.8, num_steps=50, denoised_type='dynamic_threshold', dynamic_threshold_v=99.5, sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=10)
```

# Authors

+ Arseniy Shakhmatov: [Github](https://github.com/cene555), [Blog](https://t.me/gradientdip)
+ Anton Razzhigaev: [Github](https://github.com/razzant), [Blog](https://t.me/abstractDL)
+ Aleksandr Nikolich: [Github](https://github.com/AlexWortega), [Blog](https://t.me/lovedeathtransformers)
+ Vladimir Arkhipkin: [Github](https://github.com/oriBetelgeuse)
+ Igor Pavlov: [Github](https://github.com/boomb0om)
+ Andrey Kuznetsov: [Github](https://github.com/kuznetsoffandrey)
+ Denis Dimitrov: [Github](https://github.com/denndimitrov)
