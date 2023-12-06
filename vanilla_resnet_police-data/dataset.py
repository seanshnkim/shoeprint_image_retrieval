import os
import pandas as pd
import numpy as np
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# To reproduce nearly 100% identical results across runs, this code must be inserted.
SEED = 2023
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.label_table = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.data_list = sorted(os.listdir(self.img_dir))
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        
        return sample


class TrainDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.label_table = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.data_list = sorted(os.listdir(self.img_dir))
        self.transform = transform
        self.target_transform = target_transform
        self.ref_dir = 'police_dataset/ref'
        self.ref_data_list = sorted(os.listdir(self.ref_dir))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        query_img_name, gt_ref_img_name = self.img_labels.iloc[idx, 0], self.img_labels.iloc[idx, 1]
        query_img = Image.open(os.path.join(self.img_dir, query_img_name))
        query_img = self.transform(query_img)
        
        choose_gt = random.choice(True, False)
        
        if choose_gt:
            ref_img = Image.open(os.path.join(self.ref_dir, \
                 f'{gt_ref_img_name}.png'))
            
        else:
            while True:
                random_ref_img_name = random.choice(self.ref_data_list)
                
                # FIXME: not a hard negative.
                if random_ref_img_name != gt_ref_img_name:
                    ref_img = Image.open(os.path.join(self.ref_dir, random_ref_img_name))
                    break
        
        ref_img = self.transform(ref_img)
        return query_img, ref_img, torch.from_numpy(np.array([int(choose_gt)], dtype=np.float32))