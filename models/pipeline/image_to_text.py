import os, argparse
from tqdm import tqdm
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--out_path", default="captions")
    
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = get_opt()
    
    os.makedirs(opt.out_path, exist_ok=True)
    
    image_list = os.listdir(opt.image_path)
    
    for image_name in tqdm(image_list):
        raw_image = Image.open(os.path.join(opt.image_path, image_name))
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

        out = model.generate(**inputs)
        text = processor.decode(out[0], skip_special_tokens=True)
        
        txt_name = image_name.split(".")[0] + ".txt"
        with open(os.path.join(opt.out_path, txt_name), "w") as file:
            file.write(text)