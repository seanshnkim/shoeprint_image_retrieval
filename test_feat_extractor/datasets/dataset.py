import os
import pandas as pd
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

torch.manual_seed(1000)

if torch.cuda.is_available():
    device='cuda:0'
    print('Current environment has an available GPU.')
else:
    device='cpu'
    print('Current environment does not have an available GPU.')


class DefaultDataset(Dataset):
    def __init__(self, config):
        self.data_dir = config.query_test_path
        self.data_list = os.listdir(config.query_test_path)
        self.label_table = pd.read_csv(config.image_labels)
        self.transform = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                             transforms.Resize((100,100)),
                                             transforms.ToTensor()])
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        sample_idx = int(sample.split('.')[0])
        sample_img = Image.open(os.path.join(self.data_dir, sample))
        sample_img = self.transform(sample_img).to(device)

        return sample_img, sample_idx



class TrainDataset(DefaultDataset):
    def __init__(self, config):
        super().__init__(config)
        
        self.data_dir = config.query_train_path
        self.data_list = os.listdir(self.data_dir)
        
        self.ref_dir = config.ref_train_path
        self.ref_data_list = os.listdir(self.ref_dir)
        
    def __len__(self):
        return super().__len__()
        
    def __getitem__(self, idx):
        query_img, query_idx = super().__getitem__(idx)
        gt_query_label = int(self.label_table[self.label_table.cropped == query_idx]['gt'].item())
        
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
        
        positive_ref_img = self.transform(positive_ref_img).to(device)
        negative_ref_img = self.transform(negative_ref_img).to(device)
        
        return query_img, positive_ref_img, negative_ref_img



class TestQueryDataset(DefaultDataset):
    def __init__(self, config):
        super().__init__(config)
        
        self.data_dir = config.query_test_path
        self.data_list = os.listdir(self.data_dir)
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        query_img, query_idx = super().__getitem__(idx)
        gt_query_label = int(self.label_table[self.label_table.cropped == query_idx]['gt'].item())
        
        return query_img, query_idx, gt_query_label


class TestRefDataset(DefaultDataset):
    def __init__(self, config):
        super().__init__(config)
        
        self.data_dir = config.ref_test_path
        self.data_list = os.listdir(self.data_dir)
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)