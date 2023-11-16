import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
from time import time


class Config():
    query_test_dir = "query/test"
    ref_test_dir = "ref"

    image_labels = "label_table.csv"


class TestQueryDataset(Dataset):
    def __init__(self, Config):
        self.query_dir = Config.query_test_dir
        self.query_data_list = os.listdir(self.query_dir)
        self.labeltable = pd.read_csv(Config.image_labels)
        
    def __len__(self):
        return len(self.query_data_list)
    
    def __getitem__(self, idx):
        query = self.query_data_list[idx]
        query_idx = int(query.split('.')[0])
        gt_query_label = int(self.labeltable[self.labeltable.cropped == query_idx]['gt'].item())

        return query_idx, gt_query_label

class TestRefDataset(Dataset):
    def __init__(self, Config):
        self.ref_dir = Config.ref_test_dir
        self.ref_data_list = os.listdir(self.ref_dir)
        
    def __len__(self):
        return len(self.ref_data_list)
    
    def __getitem__(self, idx):
        ref = self.ref_data_list[idx]
        ref_idx = int(ref.split('.')[0])

        return ref_idx

test_query_dataset = TestQueryDataset(Config())
test_ref_dataset = TestRefDataset(Config())

'''
ref_feats.shape: torch.Size([1175, 2048])
query_cropped_feats.shape: torch.Size([300, 2048])
query_orig_feats.shape: torch.Size([300, 2048])
'''

ref_feats = np.load("reference_features.npy")
query_cropped_feats = np.load("cropped_query_features.npy")
query_orig_feats = np.load("original_query_features.npy")
ref_feats = torch.from_numpy(ref_feats).cuda()
query_cropped_feats = torch.from_numpy(query_cropped_feats).cuda()
query_orig_feats = torch.from_numpy(query_orig_feats).cuda()

BATCH_SIZE_TEST = 1
test_query_loader = DataLoader(test_query_dataset, batch_size=1, shuffle=False)
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

labeltable = pd.read_csv(Config().image_labels)
count_1 = 0
count_5 = 0
total = 0

ONE_PCNT = int(len(test_ref_loader) * 0.01)
FIVE_PCNT = int(len(test_ref_loader) * 0.05)

logging.basicConfig(filename=f'{__file__}.log', \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    
for idx, (query_idx, gt_label) in tqdm(enumerate(test_query_loader)):
    dis_label = dict()

    top1_distance_list = list()
    top1_distance_index = list()

    top5_distance_list = list()
    top5_distance_index = list()

    with torch.inference_mode():
        # NOTE
        for i, ref_idx in enumerate(range(len(test_ref_loader))):
            euclidean_distance = F.pairwise_distance(query_cropped_feats[query_idx-1, :], ref_feats[ref_idx-1, :]).item()
            dis_label[euclidean_distance] = ref_idx
        
            if i % 500 == 0:
                del ref_idx, euclidean_distance
                gc.collect()
                torch.cuda.empty_cache()

    # Sort by distance
    sorted_dislabel=sorted(dis_label.items(),key=lambda x:x[0])

    for (i, j) in sorted_dislabel[:ONE_PCNT]:
        top1_distance_list.append(i)
        top1_distance_index.append(j)
    
    for (i, j) in sorted_dislabel[:FIVE_PCNT]:
        top5_distance_list.append(i)
        top5_distance_index.append(j)

    # print(top5_distance_list)
    # print(top5_distance_index)

    if gt_label in top1_distance_index:
        count_1 += 1
    if gt_label in top5_distance_index:
        count_5 += 1
    total += 1
        
    logging.info(f"query_number: {query_idx.item()}, gt_query_label:{gt_label.item()}")
    logging.info(f"top 1 distance index: {top1_distance_index}")
    logging.info(f"top 5 distance index: {top5_distance_index}")
    logging.info(f'count top 1% accuracy: {count_1} / total: {total}')
    logging.info(f'count top 5% accuracy: {count_5} / total: {total}\n')
    
    if idx % 10 == 0:
        del gt_label
        gc.collect()
        torch.cuda.empty_cache()