# NATALLE

![](./content/NatallE.png)
### inpainting
https://drive.google.com/file/d/1dCmCmI3jJC_LM9oFgJcn4JUoAfS2s-6D/view?usp=sharing

## 1. text2img

![](./content/bear.jpeg)

```python
from natalle.natalle_model import Natalle

model = Natalle('inference.yaml', 'natalle.pt', 'cuda', task_type='text2img')
images = model.generate_text2img('dog in the space', batch_size=2, h=512, w=512)
```
## 2. img2img
```python
from natalle.natalle_model import Natalle
from PIL import Image

model = Natalle('inference.yaml', 'natalle.pt', 'cuda', task_type='img2img')
init_image = Image.open('image.jpg')
images = model.generate_img2img('a photo of realistic landscape', init_image, strength=0.6)
```
## 3. inpainting
```python 
from natalle.natalle_model import Natalle
from PIL import Image
import numpy as np

model = Natalle('inference.yaml', 'natalle_inpainting.pt', 'cuda', task_type='inpainting')
init_image = Image.open('image.jpg')
mask = np.ones((512, 512), dtype=np.float32)
mask[100:] =  0
images = model.generate_inpainting('a red car', init_image, mask, guidance_scale=7, num_steps=50)
```
