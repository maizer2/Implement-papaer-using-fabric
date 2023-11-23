import os
import numpy as np
import torch
from torchvision.transforms import Resize
from PIL import Image

from torch.utils.data import Dataset


label_maps = [
    [128, 128, 128], # dress
    [0, 0, 128], # upper
    [0, 128, 128], # lower
    
    [0, 0, 0], # bg
    
    [0, 128, 0], # hair
    [64, 128, 128], # left arm
    [192, 128, 128], # right arm
    [192, 128, 0], # torse
    [64, 128, 128], # left hand
    [192, 128, 128], # right hand
    [64, 0, 128], # left leg
    [192, 0, 128], # right leg
    
    [0, 64, 0], # hand bag
    [192, 0, 0], # left shoes
    [64, 128, 0], # right shoes
]

dense = [
    [0, 0, 0], # background 
    
    [24, 24, 24], # head
    [2, 2, 2], # torso
    [16, 16, 16], # right inside up arm
    [18, 18, 18], # right outside up arm
    [20, 20, 20], # right inside down arm
    [22, 22, 22], # right outside down arm
    [15, 15, 15], # left inside up arm
    [17, 17, 17], # left outside up arm
    [19, 19, 19], # left inside down arm
    [21, 21, 21], # left outside down arm
    [9, 9, 9], # right up leg
    [13, 13, 13], # right down leg
    [10, 10, 10], # left up leg
    [14, 14, 14], # left down leg
    [5, 5, 5], # right foot
    [6, 6, 6], # left foot
]

parse = [
    [0, 0, 0], # background
    
    [14, 14, 14], # head
    [11 ,11, 11], # neck
    
    [5, 5, 5], # short-sleeved clothing
    [6, 6, 6], # dresses ?
    [7, 7, 7], # long_sleeved clohting
    [21, 21, 21], # left cloth arm
    [22, 22, 22], # right cloth arm
    [15, 15, 15], # left arm
    [16, 16, 16], # right arm
    
    [9, 9, 9], # left pants
    [10, 10, 10], # right pants
    [13, 13, 13], # skirt
    [17, 17, 17], # left leg
    [18, 18, 18], # right leg
    
    [19, 19, 19], # left foot
    [20, 20, 20], # right foot
    
    [23, 23, 23], # hand bag
]

class DressCode(Dataset):
    def check_data_path(self, data_path) -> os.path:
        if not (os.path.exists(data_path) and os.path.isdir(data_path)):
            raise Exception("Wrong DressCode data path")
        
        return data_path
    
    def check_cloth_type(self, cloth_type) -> str:
        if not cloth_type in {"upper_body", "lower_body", "dresses"}:
            raise Exception("Wrong cloth type.")
        
        return cloth_type
        
    def check_phase(self, phase) -> str:
        if not phase in {"train", "val", "test"}:
            raise Exception("Wrong phase.")
        
        return phase
    
    def get_pairs_name(self, paired: bool = True) -> str:
        if paired:
            return f"{self.phase}_paired.txt"
        else:
            return f"{self.phase}_unpaired.txt"
    
    def get_img_size(self, img_size: int) -> tuple:
        return (img_size, int(img_size * 0.75))
    
    def check_pairs_path(self, pairs_path) -> os.path:
        if not os.path.exists(pairs_path):
            raise Exception("Wrong pairs_list path.")
        
        return pairs_path
    
    def get_txt_lines(self, txt_path: os.path) -> list:
        with open(txt_path, "r") as f:
            lines = f.readlines()
            
        return lines
    
    def remove_enter_string(self, string) -> str:
        
        return string.split("\n")[0]
    
    def split_space_in_line(self, line) -> tuple:
        line_A, line_B = line.split(" ")
        
        return line_A, self.remove_enter_string(line_B)
    
    def split_lines_to_line(self, lines: list) -> list:
        line_list = []
        for line in lines:
            line_list.append(self.split_space_in_line(line))
        
        return line_list
    
    def get_pairs_list(self) -> list:
        pairs_path: os.path = self.check_pairs_path(os.path.join(self.data_path, self.cloth_type, self.pairs_name))
        txt_lines:  list    = self.get_txt_lines(pairs_path)
        pairs_list: list    = self.split_lines_to_line(txt_lines)
        
        return pairs_list
        
    def __init__(self,
                 data_path: str,
                 cloth_type: str = "upper_body",
                 phase: str = "train",
                 paired: bool = True,
                 img_size: int = 256):
        super().__init__()
        
        self.data_path:  os.path = self.check_data_path(data_path)
        self.cloth_type: str     = self.check_cloth_type(cloth_type)
        self.phase:      str     = self.check_phase(phase)
        self.pairs_name: str     = self.get_pairs_name(paired)
        self.img_size:   tuple   = self.get_img_size(img_size)
        
        self.pairs_list: list    = self.get_pairs_list()

        self.resize:     Resize  = Resize(self.img_size, antialias=True)
        
    def convert_path_to_numpy(self, image_path: os.path, convert_mode: str = "RGB") -> np.ndarray:
        image = Image.open(image_path).convert(convert_mode)
        image = np.array(image)
        
        return image
    
    def get_255_ndarray(self, item: np.ndarray) -> np.ndarray:
        return (item == 255).astype(np.int8)
    
    def get_parse(self, item: np.ndarray, sub_path: str) -> np.ndarray:
        item_size = (item.shape[0], item.shape[1])
        parse_list = globals()[sub_path]
        
        item_parse = np.zeros((*item_size, len(parse_list)))
        for idx, parse_part in enumerate(parse_list):
            item_parse[np.all(item == parse_part, axis=-1), idx] = 1.0
        
        return item_parse
    
    def convert_numpy_to_torch(self, ndarray: np.ndarray) -> torch.Tensor:
        torch_tensor = torch.from_numpy(ndarray)
        
        if len(torch_tensor.shape) == 2:
            torch_tensor = torch_tensor.unsqueeze(-1)

        torch_tensor = torch_tensor.permute(2, 0, 1).contiguous()
        
        return torch_tensor
    
    def pixel_value_scaling(self, image):
        return image / 255.0
    
    def mean_subtraction(self, image):
        image_mean = np.mean(image)
        
        return image - image_mean
    
    def standard_deviation_normalization(self, image):
        image_std = np.std(image)
        
        return image / image_std
    
    def get_image(self, sub_path, item_name, file_extension: str = "jpg",
                  convert_mode: str = "RGB", resize: bool = False,
                  to_tensor: bool = False, to_norm: bool = False) -> torch.Tensor:
        
        item_path = os.path.join(self.data_path, self.cloth_type, sub_path, item_name + f".{file_extension}")
        item = self.convert_path_to_numpy(item_path, convert_mode)
        
        if convert_mode == "L":
            item = self.get_255_ndarray(item)
            
        if sub_path in {"label_maps", "dense", "parse"}:
            item = self.get_parse(item, sub_path)
        
        if to_norm:
            item = self.pixel_value_scaling(item)
            item = self.mean_subtraction(item)
            item = self.standard_deviation_normalization(item)
            
        if to_tensor:
            item = self.convert_numpy_to_torch(item)
            
        if resize:
            item = self.resize(item)
            
        return item
    
    def __getitem__(self, index):
        original_name, clothing_name = self.pairs_list[index]
        original_name, clothing_name = original_name.split(".")[0], clothing_name.split(".")[0]
        
        original_tensor         = self.get_image("images",      original_name,                                              to_norm=True, to_tensor=True, resize=True)
        clothing_tensor         = self.get_image("images",      clothing_name,                                              to_norm=True, to_tensor=True, resize=True)
        clothing_mask_tensor    = self.get_image("cloth_mask",  clothing_name,                     "png", convert_mode="L",               to_tensor=True, resize=True)
        label_map_tensor        = self.get_image("label_maps",  original_name.replace("_0", "_4"), "png",                                 to_tensor=True, resize=True)
        dense_tensor            = self.get_image("dense",       original_name.replace("_0", "_5"), "png",                                 to_tensor=True, resize=True)
        parse_tensor            = self.get_image("parse",       original_name,                     "png",                                 to_tensor=True, resize=True)
        keypoints_tensor        = self.get_image("skeletons",   original_name.replace("_0", "_5"),        convert_mode="L", to_norm=True, to_tensor=True, resize=True)
        
        return {"original_name": original_name, "clohting_name": clothing_name,
                
                "original_image": original_tensor,
                "clothing_image": clothing_tensor,
                "clothing_mask": clothing_mask_tensor,
                "original_label_map": label_map_tensor,
                "original_dense": dense_tensor,
                "original_parse": parse_tensor,
                "original_keypoints": keypoints_tensor
                }
        
    def __len__(self):
        return len(self.pairs_list)
    
    def __name__(self):
        return "DressCode"
    
if __name__ == "__main__":
    dresscode = DressCode(data_path="data/DressCode")
    dresscode.__getitem__(0)