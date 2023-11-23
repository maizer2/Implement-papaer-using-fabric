import os, random, argparse


class DressCode_pairs_txt:        
    def check_ratio(self, train_ratio, validation_ratio, test_ratio):
        if not (train_ratio + validation_ratio + test_ratio == 1.0):
            raise Exception("The sum of the ratios must be 1.0.")
        
        
    def check_pairs(self,
                    original_list: list,
                    clothing_list: list):
        
        original_list.sort(), clothing_list.sort()
        
        if len(original_list) != len(clothing_list):
            raise Exception("Two list aren't match length")
        
        for idx, a in enumerate(original_list):
            if a.split("_")[0] != clothing_list[idx].split("_")[0]:
                raise Exception("Each data isn't match")
            
        return original_list, clothing_list
    
    
    def split_files(self, data_path: os.path):
        data_list = os.listdir(data_path)
        
        original_list, clothing_list = [], []
        for data in data_list:
            if "_0" in data:
                original_list.append(data)
            elif "_1" in data:
                clothing_list.append(data)
            else:
                raise Exception("Check files name.")
        
        return self.check_pairs(original_list, clothing_list)
        
    
    def get_pairs_list(self, cloth_type: str):
        if not cloth_type in {"upper_body", "lower_body", "dresses"}:
            raise Exception("Wrong cloth type.")
        
        images_path = os.path.join(self.data_path, cloth_type, "images")
        
        return self.split_files(images_path)
    
    
    def get_phase_index(self, cloth_list, train_ratio, validation_ratio) -> list:
        # 60 : 20 : 20
        total_length = len(cloth_list[0])
        
        train_index         = int(total_length * train_ratio)
        validation_index    = train_index + int(total_length * validation_ratio)
        
        return [train_index, validation_index]
        
        
    def get_index(self, train_ratio, validation_ratio):
        upper_index = self.get_phase_index(self.upper_body_list, train_ratio, validation_ratio)
        lower_index = self.get_phase_index(self.lower_body_list, train_ratio, validation_ratio)
        dress_index = self.get_phase_index(self.dresses_list, train_ratio, validation_ratio)

        return [upper_index, lower_index, dress_index]
    
    
    def __init__(self,
                 data_path: str,
                 train_ratio: float = 0.6, validation_ratio: float = 0.2, test_ratio: float = 0.2):
        self.check_ratio(train_ratio, validation_ratio, test_ratio)
        
        self.data_path = data_path
        
        self.upper_body_list = self.get_pairs_list("upper_body")
        self.lower_body_list = self.get_pairs_list("lower_body")
        self.dresses_list = self.get_pairs_list("dresses")
        
        self.index = self.get_index(train_ratio, validation_ratio)
        
        
    def split_list(self, cloth_list, phase_index):
        train_list = [cloth_list[0][:phase_index[0]], cloth_list[1][:phase_index[0]]]
        validation_list = [cloth_list[0][phase_index[0]:phase_index[1]], cloth_list[1][phase_index[0]:phase_index[1]]]
        test_list = [cloth_list[0][phase_index[1]:], cloth_list[1][phase_index[1]:]]

        return train_list, validation_list, test_list
    
    
    def get_phase_pairs_txt(self, cloth_list, paired: bool = True):
        original, clothing = cloth_list
        
        if not paired:
            random.shuffle(clothing)
            
        txt = ""
        for idx, item in enumerate(original):
            txt += f"{item} {clothing[idx]}\n"
            
        return txt
    
    
    def get_pairs_txt(self, cloth_list, paired: bool = True) -> str:
        train_pairs = self.get_phase_pairs_txt(cloth_list[0], paired)
        val_pairs = self.get_phase_pairs_txt(cloth_list[1], paired)
        test_pairs = self.get_phase_pairs_txt(cloth_list[2], paired)
    
        return train_pairs, val_pairs, test_pairs
    
    
    def write_txt(self, txt: list, phase: str, paired: bool, cloth_type: str):
        if paired:
            paired_name = "paired.txt"
        else:
            paired_name = "unpaired.txt"
            
        out_path = os.path.join(self.data_path, cloth_type, phase + "_" + paired_name)
        
        with open(out_path, "w") as f:
            f.write(txt)
            
    
    def write_pairs_txt(self, cloth_list: list, paired: bool, cloth_type: str):
        paired_txt = self.get_pairs_txt(cloth_list, paired)
        
        self.write_txt(paired_txt[0], "train", paired, cloth_type)
        self.write_txt(paired_txt[1], "val", paired, cloth_type)
        self.write_txt(paired_txt[2], "test", paired, cloth_type)
        
        
    def make_pairs_list(self, cloth_list: list, cloth_type: str):
        self.write_pairs_txt(cloth_list, True, cloth_type)
        self.write_pairs_txt(cloth_list, False, cloth_type)
            
    
    
    def __call__(self):
        upper_list = self.split_list(self.upper_body_list, self.index[0])
        lower_list = self.split_list(self.lower_body_list, self.index[1])
        dress_list = self.split_list(self.dresses_list, self.index[2])
        
        self.make_pairs_list(upper_list, "upper_body")
        self.make_pairs_list(lower_list, "lower_body")
        self.make_pairs_list(dress_list, "dresses")


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/DressCode/")
    
    parser.add_argument("--train_ratio", default=0.6)
    parser.add_argument("--validation_ratio", default=0.2)
    parser.add_argument("--test_ratio", default=0.2)
    opt = parser.parse_args()
    
    return opt


if __name__ == "__main__":
    opt = get_opt()
    
    cls = DressCode_pairs_txt(opt.data_path,
                              opt.train_ratio, opt.validation_ratio, opt.test_ratio)
    cls()