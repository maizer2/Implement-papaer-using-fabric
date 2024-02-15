'''
Code for separating clothing from garment images used by Virtual Try-on Network.
This requires an additional human-parse image.
The human-parse image you can get it from "https://github.com/GoGoDuck912/Self-Correction-Human-Parsing".
'''

import argparse, os
from PIL import Image

class seperate_body_and_garment:
    def __init__(self,
                 cloth_path,
                 parse_path,
                 out_path=None,
                 error_log: bool = True):
        
        self.error_log = error_log
        
        self.cloth_path = cloth_path
        self.parse_path = parse_path
        self.out_path = out_path if out_path is not None else "./only_garment"
        self.error_log = error_log
        
        self.cloth_list = os.listdir(cloth_path)
        self.parse_list = os.listdir(parse_path)
        
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self):
        for cloth in self.cloth_list:
            parse = Image.open(os.path.join(self.parse_path, cloth.replace("jpg", "png")))
            
            

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloth_path", type=str, required=True,
                        help="garment image path.")
    parser.add_argument("--parse_path", type=str, required=True,
                        help="garment parse image path.")
    parser.add_argument("--out_path", type=str, required=False,
                        help="output path.")
    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = get_opt()
    
    cls = seperate_body_and_garment(cloth_path=opt.cloth_path,
                                    parse_path=opt.parse_path,
                                    out_path=opt.out_path)
    cls()