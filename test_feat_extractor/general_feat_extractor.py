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

from dataset import TestFeatDataset
from utils import set_device

device = set_device()

'''
python test_feat_extractor/test_feat_extractor.py
    --model_name msn_large \
    --feat_combined True \
    --query test_feat_extractor/np_features_naver/cropped_query_features.npy \
    --reference test_feat_extractor/np_features_naver/reference_features.npy \
    
python test_feat_extractor/test_feat_extractor.py
    --feat_combined False \
    --query_dir query/test \
    --reference ref \
'''

parser = argparse.ArgumentParser(
        description='Test feature extractor (returns top 1%, 5% accuracy from query and reference features)',
        epilog='python test_feat_extractor/test_fetop5_distance_index --ref_dir ref')
parser.add_argument('--model_name', type=str, default='msn_large', help='model name')
parser.add_argument('--feat_combined', type=str, default="True", \
                    help='True if query and reference features are combined in a single numpy file')
parser.add_argument('--query', type=str, \
    help='if features are combined, write numpy file path. If features are separate, write down query feature path')
parser.add_argument('--reference', type=str, \
    help='if features are combined, write numpy file path. If features are separate, write down query feature path')

args = parser.parse_args()

default_working_dir = 'test_feat_extractor'
with open(os.path.join(default_working_dir, 'config.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)

if args.feat_combined == 'True':
    ref_feats = torch.from_numpy(np.load(args.reference)).cuda()
    query_feats = torch.from_numpy(np.load(args.query)).cuda()
else:
    # stack all the numpy files in args.query_dir
    query_feats, ref_feats = [], []

BATCH_SIZE_QUERY = 1
BATCH_SIZE_REF = 128
test_query_dataset = TestQueryDataset(cfg)
test_ref_dataset = TestRefDataset(cfg)
test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE_QUERY, shuffle=True)
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_REF, shuffle=True)

ONE_PCNT = int(len(test_ref_dataset) * 0.01)
FIVE_PCNT = int(len(test_ref_dataset) * 0.05)
TOP_50 = 50

if __name__ == "__main__":
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    log_save_dir = os.path.join(default_working_dir, 'logs', f'test_feat_extractor_{start_time_stamp}.log')
    logging.basicConfig(filename=f'{log_save_dir}.log', \
            level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    count_1 = 0
    count_5 = 0
    total = 0
    
    for qidx, (_, query_idx, gt_label) in tqdm(enumerate(test_query_loader)):
        dist_label_tuple = []

        with torch.inference_mode():
            for i, (_, ref_label) in enumerate(test_ref_loader):
                if args.feat_combined == 'True':
                    euclidean_distance = F.pairwise_distance(query_feats[query_idx-1, :], ref_feats[ref_label-1, :])
                    euclidean_distance = euclidean_distance.cpu().detach().numpy()
                else:
                    
                
                for j in range(len(euclidean_distance)):
                    dist_label_tuple.append((euclidean_distance[j], ref_label[j].item()) )
        
        tup_sorted = sorted(dist_label_tuple, key=lambda x: x[0])
        top1pct_list = tup_sorted[:ONE_PCNT]
        top50_list = tup_sorted[:TOP_50]
        
        top1pct_distance_index = [i[1] for i in top1pct_list]
        top50_distance_index = [i[1] for i in top50_list]
        
        if gt_label in top1pct_distance_index:
            count_1 += 1
        if gt_label in top50_distance_index:
            count_5 += 1
        total += 1
        
        logging.info(f"query_number: {query_idx.item()}, gt_query_label:{gt_label.item()}")
        logging.info(f"top 1% distance index: {top1pct_distance_index}")
        logging.info(f"top 50 distance index: {top50_distance_index}")
        logging.info(f'count top 1% accuracy: {count_1} / total: {total}')
        logging.info(f'count top 50 accuracy: {count_5} / total: {total}\n')
        
        if qidx % 25 == 0:
            del gt_label
            gc.collect()
            torch.cuda.empty_cache()
    
    logging.info(f'Final top 1% accuracy: {(count_1 / total) * 100} %')
    logging.info(f'FINAL top 50 accuracy: {(count_5 / total) * 100} %')