import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "a close up of a denim shirt with a v on the chest"
image = Image.open("data/DressCode/upper_body/images/000000_0.jpg").resize((512, 384))
mask_image = Image.open("inpaint_mask.png").resize((512, 384))
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./000000_0_on_000001_1.png")
