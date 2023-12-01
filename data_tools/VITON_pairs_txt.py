import os, random, argparse


class VITON_pairs_txt:        
    def check_ratio(self, train_ratio, validation_ratio, test_ratio):
        if not (train_ratio + validation_ratio + test_ratio == 1.0):
            raise Exception("The sum of the ratios must be 1.0.")
             
    def check_pairs(self, data_path):
        image_list = os.listdir(os.path.join(data_path, "image"))
        cloth_list = os.listdir(os.path.join(data_path, "cloth"))
        
        image_list.sort(), cloth_list.sort()
        
        if len(image_list) != len(cloth_list):
            raise Exception("Two list aren't match length")
        
        for idx, a in enumerate(image_list):
            if a.split("_")[0] != cloth_list[idx].split("_")[0]:
                raise Exception("Each data isn't match")
            
        return image_list, cloth_list
        
    def get_phase_index(self, train_ratio, validation_ratio) -> list:
        # 60 : 20 : 20
        total_length = len(self.image_list)
        
        train_index         = int(total_length * train_ratio)
        validation_index    = train_index + int(total_length * validation_ratio)
        
        return [train_index, validation_index]
        
    def __init__(self,
                 data_path: str,
                 train_ratio: float = 0.6, validation_ratio: float = 0.2, test_ratio: float = 0.2):
        self.data_path = data_path
        self.check_ratio(train_ratio, validation_ratio, test_ratio)
        
        self.image_list, self.cloth_list = self.check_pairs(data_path)
        
        self.index = self.get_phase_index(train_ratio, validation_ratio)
        
    def split_list(self):
        train_list = [self.image_list[:self.index[0]], self.cloth_list[:self.index[0]]]
        validation_list = [self.image_list[self.index[0]:self.index[1]], self.cloth_list[self.index[0]:self.index[1]]]
        test_list = [self.image_list[self.index[1]:], self.cloth_list[self.index[1]:]]

        return train_list, validation_list, test_list
    
    def get_phase_pairs_txt(self, cloth_list, paired: bool = True):
        original, clothing = cloth_list
        
        if not paired:
            random.shuffle(clothing)
            
        txt = ""
        for idx, item in enumerate(original):
            txt += f"{item} {clothing[idx]}\n"
            
        return txt
    
    def get_pairs_txt(self, data_list, paired: bool = True) -> str:
        train_pairs = self.get_phase_pairs_txt(data_list[0], paired)
        val_pairs = self.get_phase_pairs_txt(data_list[1], paired)
        test_pairs = self.get_phase_pairs_txt(data_list[2], paired)
    
        return train_pairs, val_pairs, test_pairs
    
    def write_txt(self, txt: list, phase: str, paired: bool):
        if paired:
            paired_name = "paired.txt"
        else:
            paired_name = "unpaired.txt"
            
        out_path = os.path.join(self.data_path, phase + "_" + paired_name)
        
        with open(out_path, "w") as f:
            f.write(txt)
        
    def write_pairs_txt(self, data_list: list, paired: bool):
        paired_txt = self.get_pairs_txt(data_list, paired)
        
        self.write_txt(paired_txt[0], "train", paired)
        self.write_txt(paired_txt[1], "val", paired)
        self.write_txt(paired_txt[2], "test", paired)
        
    def make_pairs_list(self, data_list: list):
        self.write_pairs_txt(data_list, True)
        self.write_pairs_txt(data_list, False)
            
    def __call__(self):
        data_list = self.split_list()
        self.make_pairs_list(data_list)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/VITON/")
    
    parser.add_argument("--train_ratio", default=0.6)
    parser.add_argument("--validation_ratio", default=0.2)
    parser.add_argument("--test_ratio", default=0.2)
    opt = parser.parse_args()
    
    return opt


if __name__ == "__main__":
    opt = get_opt()
    
    VITON_pairs_txt(opt.data_path, opt.train_ratio, opt.validation_ratio, opt.test_ratio)()
