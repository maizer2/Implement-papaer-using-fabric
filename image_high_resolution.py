import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# let's download an  image
url = "04136.jpg"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = Image.open(url).convert("RGB")
original_img_size = low_res_img.size

low_res_img = low_res_img.resize((128, 128))

# run pipeline in inference (sample random noise and denoise)
upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
upscaled_image = upscaled_image.resize((original_img_size[0]*4, original_img_size[1]*4))
# save image
upscaled_image.save("ldm_generated_image.png")