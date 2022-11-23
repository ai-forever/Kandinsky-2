# Kandinsky 2.0

![](./content/NatallE.png)

## 1. text2img

![](./content/bear.jpeg)

```python
from kandinsky2 import get_kandinsky2

model = get_kandinsky2('cuda', task_type='text2img')
images = model.generate_text2img('кошка', batch_size=4, h=512, w=512, num_steps=75, denoised_type='dynamic_threshold', dynamic_threshold_v=99.5, sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=10)
```
## 2. img2img
```python
from kandinsky2 import get_kandinsky2
from PIL import Image

model = get_kandinsky2('cuda', task_type='text2img')
init_image = Image.open('image.jpg')
images = model.generate_img2img('кошка',  init_image, strength=0.8, num_steps=50, denoised_type='dynamic_threshold', dynamic_threshold_v=99.5, sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=10)
```
## 3. inpainting
```python 
from kandinsky2 import get_kandinsky2
from PIL import Image
import numpy as np

model = get_kandinsky2('cuda', task_type='inpainting')
init_image = Image.open('image.jpg')
mask = np.ones((512, 512), dtype=np.float32)
mask[100:] =  0
images = model.generate_inpainting('красная кошка', init_image, mask, num_steps=50, denoised_type='dynamic_threshold', dynamic_threshold_v=99.5, sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=10)
```
