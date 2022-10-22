# NATALL-E
## 1. text2img
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
