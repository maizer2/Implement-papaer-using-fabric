from diffusers import StableDiffusionInpaintPipeline

import torch
from PIL import Image

image = Image.open("overture-creations-5sI6fQgYIuo.png")
mask_image = Image.open("overture-creations-5sI6fQgYIuo_mask.png")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./overture-creations-5sI6fQgYIuo_output.png")