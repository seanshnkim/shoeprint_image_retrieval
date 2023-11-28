import os
import numpy as np
from tqdm import tqdm
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import logging
from time import strftime, localtime
import yaml
import argparse

from dataset import FeatQueryDataset, FeatRefDataset
from utils import set_device


device = set_device()
default_working_dir = "test_feat_extractor"
with open(os.path.join(default_working_dir, 'config.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='None', help='feature extractor model')
args = parser.parse_args()

if cfg[args.model_name]["feat_combined"]:
    ref_feats_combined = np.load(cfg[args.model_name]["ref"])
    query_feats_combined = np.load(cfg[args.model_name]["query"])
    
    ref_save_dir = os.path.join(os.path.dirname(cfg[args.model_name]["ref"]), 'ref')
    query_save_dir = os.path.join(os.path.dirname(cfg[args.model_name]["query"]), 'query')
    os.makedirs(ref_save_dir, exist_ok=True)
    os.makedirs(query_save_dir, exist_ok=True)
    
    for i in range(len(ref_feats_combined)):
        if os.path.exists(os.path.join(ref_save_dir, f'{i+1:05d}.npy')):
            continue
        np.save(os.path.join(ref_save_dir, f'{i+1:05d}.npy'), ref_feats_combined[i])
    for i in range(len(query_feats_combined)):
        if os.path.exists(os.path.join(query_save_dir, f'{i+1:05d}.npy')):
            continue
        np.save(os.path.join(query_save_dir, f'{i+1:05d}.npy'), query_feats_combined[i])
        
    cfg[args.model_name]["ref"] = ref_save_dir
    cfg[args.model_name]["query"] = query_save_dir


BATCH_SIZE_QUERY = 1
BATCH_SIZE_REF = 128
test_query_dataset = FeatQueryDataset(cfg, cfg[args.model_name]["query"])
test_ref_dataset  = FeatRefDataset(cfg, cfg[args.model_name]["ref"])
test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE_QUERY, shuffle=True)
test_ref_loader = DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_REF, shuffle=True)

ONE_PCNT = int(len(test_ref_dataset) * 0.01)
FIVE_PCNT = int(len(test_ref_dataset) * 0.05)
TOP_50 = 50

if __name__ == "__main__":
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    cur_fname = os.path.basename(__file__).rstrip('.py')
    log_save_dir = os.path.join(default_working_dir, 'logs', f'{cur_fname}_{start_time_stamp}.log')
    logging.basicConfig(filename=log_save_dir, \
            level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logging.info(f"model_name: {args.model_name}")
    
    count_1 = 0
    count_5 = 0
    total = 0
    
    for qidx, (query_feat, query_idx, gt_label) in tqdm(enumerate(test_query_loader)):
        dist_label_tuple = []

        with torch.inference_mode():
            for i, (ref_feats, ref_indices) in enumerate(test_ref_loader):
                # if model == msn_large, ref_feats.shape = (128, 197, 1024) query_feats.shape = (1, 197, 1024)
                # euclidean_distance = F.pairwise_distance(query_feat, ref_feats)
                # euclidean_distance = euclidean_distance.cpu().detach().numpy()
                euclidean_distances = torch.sqrt(torch.sum((ref_feats - query_feat) ** 2, dim=-1)).mean(dim=-1)

                for j in range(len(euclidean_distances)):
                    dist_label_tuple.append((euclidean_distances[j], ref_indices[j].item()) )
    
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