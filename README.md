# NATALL-E
## 1. text2img
`from natalle.natalle_model import Natalle
model = Natalle('inference.yaml', 'natalle.pt', 'cuda', task_type='text2img')
images = model.generate_text2img('dog in the space', batch_size=2, h=512, w=512)`
