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

class DefaultDataset(Dataset):
    def __init__(self, config):
        self.data_dir = os.path.join(config["working_dir"], config["query_test_path"])
        self.data_list = sorted(os.listdir(self.data_dir))
        self.label_table = pd.read_csv(os.path.join(config["working_dir"], config["image_labels"]))
        self.transform = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                             transforms.Resize((100,100)),
                                             transforms.ToTensor()])
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        sample_idx = int(sample.split('.')[0])
        sample_img = Image.open(os.path.join(self.data_dir, sample))
        sample_img = self.transform(sample_img)

        return sample_img, sample_idx



class TrainDataset(DefaultDataset):
    def __init__(self, config, loss_function):
        super().__init__(config)
        
        self.loss_function = loss_function
        self.data_dir = os.path.join(config["working_dir"], config["query_train_path"])
        self.data_list = sorted(os.listdir(self.data_dir))
        
        self.ref_dir = os.path.join(config["working_dir"], config["ref_train_path"])
        self.ref_data_list = sorted(os.listdir(self.ref_dir))
        
    def __len__(self):
        return super().__len__()
    
        
    def __getitem__(self, idx):
        query_img, query_idx = super().__getitem__(idx)
        gt_query_label = int(self.label_table[self.label_table.cropped == query_idx]['gt'].item())
        
        #FIXME - Needs to be refactored
        if self.loss_function == 'contrastive':
            should_get_same_class = random.randint(0, 1)
            
            if should_get_same_class:
                ref_version = random.choice([-1, 0, 1])
                if ref_version == -1:
                    ref_img = Image.open(os.path.join(self.ref_dir, \
                        '{0:05d}'.format(gt_query_label) +'.png'))
                else:
                    ref_img = Image.open(os.path.join(self.ref_dir, \
                        '{0:05d}'.format(gt_query_label) + '_' + str(ref_version) + '.jpg'))
                
                ref_label = gt_query_label
                
            else:
                while True:
                    ref = random.choice(self.ref_data_list)
                    ref_label = int(ref.split('.')[0].split('_')[0])
                    if gt_query_label != ref_label:
                        ref_img = Image.open(os.path.join(self.ref_dir, ref))
                        break
            
            ref_img = self.transform(ref_img)
            
            return query_img, ref_img, torch.from_numpy(np.array([int(gt_query_label != ref_label)],dtype=np.float32))

        # If triplet loss
        elif self.loss_function == 'triplet':
            ref_version = random.choice([-1, 0, 1])
            if ref_version == -1:
                positive_ref_img = Image.open(os.path.join(self.ref_dir, \
                    '{0:05d}'.format(gt_query_label) +'.png'))
            else:
                positive_ref_img = Image.open(os.path.join(self.ref_dir, \
                    '{0:05d}'.format(gt_query_label) + '_' + str(ref_version) + '.jpg'))
            
            while True:
                ref = random.choice(self.ref_data_list)
                negative_ref_label = int(ref.split('.')[0].split('_')[0])
                if gt_query_label != negative_ref_label:
                    negative_ref_img = Image.open(os.path.join(self.ref_dir, ref))
                    break
            
            positive_ref_img = self.transform(positive_ref_img)
            negative_ref_img = self.transform(negative_ref_img)
            
            return query_img, positive_ref_img, negative_ref_img

        else:
            raise NotImplementedError


class TestQueryDataset(DefaultDataset):
    def __init__(self, config):
        super().__init__(config)
        
        self.data_dir = os.path.join(config["working_dir"], config["query_test_path"])
        self.data_list = sorted(os.listdir(self.data_dir))
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        query_img, query_idx = super().__getitem__(idx)
        gt_query_label = int(self.label_table[self.label_table.cropped == query_idx]['gt'].item())
        
        return query_img, query_idx, gt_query_label


class TestRefDataset(DefaultDataset):
    def __init__(self, config):
        super().__init__(config)
        
        self.data_dir = os.path.join(config["working_dir"], config["ref_test_path"])
        self.data_list = sorted(os.listdir(self.data_dir))
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)