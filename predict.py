from cog import BasePredictor, Input, Path
from kandinsky2 import get_kandinsky2

CACHE_DIR = "kandinsky2-weights"


class Predictor(BasePredictor):
    def setup(self):
        self.model = get_kandinsky2(
            "cuda",
            task_type="text2img",
            cache_dir="/tmp/kandinsky2",
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
    ) -> Path:
        out = "/tmp/out.png"

        images = self.model.generate_text2img(
            prompt,
            num_steps=num_inference_steps,
            batch_size=1,
            guidance_scale=guidance_scale,
            h=768,
            w=768,
            sampler=scheduler,
            prior_cf_scale=prior_cf_scale,
            prior_steps=prior_steps,
        )
        images[0].save(out)
        return Path(out)
