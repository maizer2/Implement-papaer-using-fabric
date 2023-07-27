#coding=utf-8
from typing import Any
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os
import numpy as np
import json

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.data_path
        self.datamode = "train" # train or test or self-defined
        self.stage = "TOM"
        self.data_list = f"train_unpairs.txt"
        self.img_size = opt.image_size
        self.radius = 5
        self.data_path = os.path.join(self.root, self.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=opt.image_mean, 
                std=opt.image_std
            )
        ])

        # load data list
        im_names = []
        c_names = []

        with open(os.path.join(self.root, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        # cloth
        c_name = self.c_names[index]
        # target
        im_name = self.im_names[index]

        # cloth image & cloth mask
        c = Image.open(os.path.join(self.data_path, 'warp-cloth', c_name))
        cm = Image.open(os.path.join(self.data_path, 'warp-mask', c_name))
        im_g = ''
            
        # 256x192x3 -> 3x256x192
        c = self.transform(c)
        if (c.shape[1], c.shape[2]) != self.img_size:
            c = transforms.Resize(self.img_size)(c)

        # 1x256x192 -> 256x192
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        # 256x192 -> 1x256x192
        cm = torch.from_numpy(cm_array).unsqueeze(0)

        # person image
        im = Image.open(os.path.join(self.data_path, 'image', im_name))
        # 256x192x3 -> 3x256x192
        im = self.transform(im)

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(os.path.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
       
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.img_size[1]//16, self.img_size[0]//16))
        parse_shape = parse_shape.resize((self.img_size[1], self.img_size[0]))
        
        shape = self.transform(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, *self.img_size)
        
        r = self.radius
        im_pose = Image.new('L', (self.img_size[1], self.img_size[0]))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.img_size[1], self.img_size[0]))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform(im_pose)

        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0) 
        
        
        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        self.dataset = dataset
        self.data_loader = data.DataLoader(
                dataset, 
                batch_size=opt.batch_size,
                num_workers=opt.num_workers, 
                )
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
    
    def get_loader(self):
        return self.data_loader