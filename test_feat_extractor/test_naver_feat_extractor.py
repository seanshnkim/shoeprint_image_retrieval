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


# reference 이미지를 폴더에서 뱉는 데이터셋
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

labeltable = pd.read_csv(Config().imagelabels)
count_1 = 0
count_5 = 0
total = 0

ONE_PCNT = int(len(test_ref_loader) * 0.01)
FIVE_PCNT = int(len(test_ref_loader) * 0.05)

logging.basicConfig(filename=f'{__file__}.log', \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
for idx, (query_idx, gt_label) in tqdm(enumerate(test_query_loader)):
    # test_query_loader에서 201 ~ 300번 사이의 test 이미지 불러오기

    dis_label = dict()

    top1_distance_list = list()
    top1_distance_index = list()

    top5_distance_list = list()
    top5_distance_index = list()
    
    logging.info(f"{idx}th query: START calculating distance")

    calc_dist_t = 0
    clean_t = 0
    with torch.inference_mode():
        # NOTE
        for i, ref_idx in enumerate(range(len(test_ref_loader))):
            start_t = time()
            euclidean_distance = F.pairwise_distance(query_cropped_feats[query_idx-1, :], ref_feats[ref_idx-1, :]).item()
            dis_label[euclidean_distance] = ref_idx
        
            middle_t = time()
            calc_dist_t += middle_t - start_t

            end_t = time()
            clean_t += end_t - middle_t
            
            if i % 100 == 0:
                del ref_idx, euclidean_distance
                gc.collect()
                torch.cuda.empty_cache()
                
                logging.info(f"{idx}th query: {i}th reference: {calc_dist_t} sec for calculating distance, {clean_t} sec for cleaning")
                calc_dist_t = 0
                clean_t = 0
                
    logging.info(f"{idx}th query: COMPLETE calculating distance")
    
    # Sort by distance
    sorted_dislabel=sorted(dis_label.items(),key=lambda x:x[0])

    for (i, j) in sorted_dislabel[:ONE_PCNT]:
        top1_distance_list.append(i)
        top1_distance_index.append(j)
    
    for (i, j) in sorted_dislabel[:FIVE_PCNT]:
        top5_distance_list.append(i)
        top5_distance_index.append(j)
    print("query_number: {}, gt_query_label:{}".format(query_idx, gt_label))

    print(top1_distance_list)
    print(top1_distance_index)

    print(top5_distance_list)
    print(top5_distance_index)

    if gt_label in top1_distance_index : count_1 += 1
    if gt_label in top5_distance_index : count_5 += 1
    total += 1

    print('count_1%: {} / total: {}'.format(count_1, total))
    print('count_5%: {} / total: {}'.format(count_5, total))
        
    del gt_label
    
    gc.collect()
    torch.cuda.empty_cache()
    
    logging.info(f"{idx}th query: COMPLETE cleaning")