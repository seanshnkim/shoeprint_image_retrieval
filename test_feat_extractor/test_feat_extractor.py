import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import logging
from time import strftime, localtime
import yaml
import argparse
import heapq

from dataset import TestQueryDataset, TestRefDataset

'''
python test_feat_extractor/test_feat_extractor.py
    --model_name msn_large \
    --mode combined \
    --query test_feat_extractor/np_features_naver/cropped_query_features.npy \
    --ref test_feat_extractor/np_features_naver/reference_features.npy \
    
python test_feat_extractor/test_feat_extractor.py
    --mode separate \
    --query_dir query/test \
    --ref_dir ref \
'''

parser = argparse.ArgumentParser(
        description='Test feature extractor (returns top 1%, 5% accuracy from query and reference features)',
        epilog='python test_feat_extractor/test_fetop5_distance_index --ref_dir ref')

default_working_dir = 'test_feat_extractor'
with open(os.path.join(default_working_dir, 'config.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)

start_time_stamp = strftime("%m-%d_%H%M", localtime())
log_save_dir = os.path.join(default_working_dir, 'logs', f'test_feat_extractor_{start_time_stamp}.log')
logging.basicConfig(filename=log_save_dir, \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
ref_feats = torch.from_numpy(np.load("reference_features.npy")).cuda()
query_cropped_feats = torch.from_numpy(np.load("cropped_query_features.npy")).cuda()
query_orig_feats = torch.from_numpy(np.load("original_query_features.npy")).cuda()

BATCH_SIZE_TEST = 1
test_query_dataset = TestQueryDataset(cfg)
test_ref_dataset = TestRefDataset(cfg)
test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

ONE_PCNT = int(len(test_ref_loader) * 0.01)
# FIVE_PCNT = int(len(test_ref_loader) * 0.05)
FIVE_PCNT = 50

if __name__ == "__main__":
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    log_save_dir = os.path.join(default_working_dir, 'logs', f'test_feat_extractor_{start_time_stamp}.log')
    logging.basicConfig(filename=f'{__file__}.log', \
            level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    
    count_1 = 0
    count_5 = 0
    total = 0
    
    for qidx, (_, query_idx, gt_label) in tqdm(enumerate(test_query_loader)):
        dis_label = []

        with torch.inference_mode():
            # NOTE
            for i, (_, ref_label) in enumerate(test_ref_loader):
                euclidean_distance = F.pairwise_distance(query_cropped_feats[query_idx-1, :], ref_feats[ref_idx-1, :]).item()
                heapq.heappush(dis_label, (euclidean_distance, ref_label.item()) )
            
                if i % 500 == 0:
                    del ref_label, euclidean_distance
                    gc.collect()
                    torch.cuda.empty_cache()

        top1_list = heapq.nsmallest(ONE_PCNT, dis_label)
        top1_distance_list, top1_distance_index = [i[0] for i in top1_list], [i[1] for i in top1_list]
        
        top50_list = heapq.nsmallest(FIVE_PCNT, dis_label)
        top50_distance_list, top50_distance_index = [i[0] for i in top50_list], [i[1] for i in top50_list]

        if gt_label in top1_distance_index:
            count_1 += 1
        if gt_label in top50_distance_index:
            count_5 += 1
        total += 1
            
        logging.info(f"query_number: {query_idx.item()}, gt_query_label:{gt_label.item()}")
        logging.info(f"top 1 distance index: {top1_distance_index}")
        logging.info(f"top 50 distance index: {top50_distance_index}")
        logging.info(f'count top 1% accuracy: {count_1} / total: {total}')
        logging.info(f'count top 50 accuracy: {count_5} / total: {total}\n')
        
        if qidx % 25 == 0:
            del gt_label
            gc.collect()
            torch.cuda.empty_cache()