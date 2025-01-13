from diffusers import StableDiffusionPipeline
import torch

class ImageGenerator:
    def __init__(self):
        self.model_id = "stabilityai/stable-diffusion-2-1"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            
    def generate_image(self, prompt, num_images=1):
        images = self.pipe(
            prompt,
            num_images_per_prompt=num_images,
            guidance_scale=7.5,
            num_inference_steps=50
        ).images
        return images
