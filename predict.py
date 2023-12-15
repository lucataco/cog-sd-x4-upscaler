# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import time
import torch
from PIL import Image
from weights_downloader import WeightsDownloader
from diffusers import StableDiffusionUpscalePipeline

MODEL_NAME = "stabilityai/stable-diffusion-x4-upscaler"
MODEL_CACHE = "model-cache"
MODEL_URL = "https://weights.replicate.delivery/default/stabilityai/sd-x4-upscaler.tar"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        WeightsDownloader.download_if_not_exists(MODEL_URL, MODEL_CACHE)
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        )
        self.pipeline = pipeline.to("cuda")
        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(description="Input prompt", default="A white cat"),
        scale: int = Input(
            description="Factor to scale image by", default=4, 
            choices = [1, 2, 4]
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        pil_image = Image.open(image).convert("RGB")
        w,h = pil_image.size
        if scale == 1:
            low_res_img = pil_image.resize((w//4, h//4))
        elif scale == 2:
            low_res_img = pil_image.resize((w//2, h//2))
        else:
            low_res_img = pil_image
        print("Downscaled image size: ", low_res_img.size)

        upscaled_image = self.pipeline(
            prompt=prompt,
            image=pil_image
        ).images[0]

        output_path = "/tmp/upscaled.png"
        upscaled_image.save(output_path)
        return Path(output_path)
