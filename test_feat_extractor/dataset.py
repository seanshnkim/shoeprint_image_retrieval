import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class FeatQueryDataset(Dataset):
    def __init__(self, config, feat_path):
        self.feat_path = feat_path
        self.data_list = sorted(os.listdir(feat_path))
        self.label_table = pd.read_csv(config["image_labels"])
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        feat = np.load(os.path.join(self.feat_path, self.data_list[idx]))
        feat_idx = int(self.data_list[idx].split('.')[0])
        gt_label = int(self.label_table[self.label_table.cropped == feat_idx]['gt'].item())
        
        return feat, feat_idx, gt_label



class FeatRefDataset(Dataset):
    def __init__(self, config, feat_path):
        self.feat_path = feat_path
        self.data_list = sorted(os.listdir(feat_path))
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        feat = np.load(os.path.join(self.feat_path, self.data_list[idx]))
        feat_idx = int(self.data_list[idx].split('.')[0])
        
        return feat, feat_idx