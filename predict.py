from typing import List
from cog import BasePredictor, Input, Path
from kandinsky2 import get_kandinsky2


class Predictor(BasePredictor):
    def setup(self):
        self.model = get_kandinsky2(
            "cuda",
            task_type="text2img",
            cache_dir="./kandinsky2-weights",
            model_version="2.1",
            use_flash_attention=False,
        )

    def predict(
        self,
        prompt: str = Input(description="Input Prompt", default="red cat, 4k photo"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=4
        ),
        scheduler: str = Input(
            description="Choose a scheduler",
            default="p_sampler",
            choices=["ddim_sampler", "p_sampler", "plms_sampler"],
        ),
        prior_cf_scale: int = Input(default=4),
        prior_steps: str = Input(default="5"),
        width: int = Input(
            description="Choose width. Lower the setting if out of memory.",
            default=512,
            choices=[256, 288, 432, 512, 576, 768, 1024],
        ),
        height: int = Input(
            description="Choose height. Lower the setting if out of memory.",
            default=512,
            choices=[256, 288, 432, 512, 576, 768, 1024],
        ),
        batch_size: int = Input(
            description="Choose batch size. Lower the setting if out of memory.",
            default=1,
            choices=[1, 2, 3, 4],
        ),
    ) -> List[Path]:
        images = self.model.generate_text2img(
            prompt,
            num_steps=num_inference_steps,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=height,
            w=width,
            sampler=scheduler,
            prior_cf_scale=prior_cf_scale,
            prior_steps=prior_steps,
        )
        output = []
        for i, im in enumerate(images):
            out = f"/tmp/out_{i}.png"
            im.save(out)
            im.save(f"out_{i}.png")
            output.append(Path(out))
        return output
