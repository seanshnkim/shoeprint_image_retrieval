import numpy as np
from tqdm import tqdm
from time import strftime, localtime
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

import logging

from config import Config
from loss import ContrastiveLoss
from dataset import TestQueryDataset, TestRefDataset
from utils import set_device

device = set_device()
torch.manual_seed(1000)

# Choose loss function
loss_type = "triplet" # contrastive or triplet
# choose which model ckpt to load
ckpt_path = ""

cfg = Config(loss="loss_type", working_dir="test_feat_extractor")
model_hps = cfg.get_model_hyperparameters()
train_hps = cfg.get_training_hyperparameters()

save = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            save.append(obj.size())
    except:
        pass
print(np.unique(save, return_counts=True))

test_query_dataset = TestQueryDataset(cfg)
test_ref_dataset = TestRefDataset(cfg)
BATCH_SIZE_TEST = 1
test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

ONE_PCNT = int(len(test_ref_loader) * 0.01)
FIVE_PCNT = int(len(test_ref_loader) * 0.05)

    
if __name__ == "__main__":
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    log_save_dir = os.path.join(cfg.working_dir, 'logs', f'{loss_type}_test_{start_time_stamp}.log')
    logging.basicConfig(filename=log_save_dir, \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logging.info(f"Model Hyperparameters: {model_hps}\n")
    logging.info(f"Training Hyperparameters: {train_hps}\n")

    if cfg.loss_function == "triplet":
        loss_fn = nn.TripletMarginLoss(margin=train_hps["margin"], p=2)
    else:
        loss_fn = ContrastiveLoss(margin=train_hps["margin"])
    
    model = torch.load("triplet_05-06_2205.pt")
    model.eval()
    count_1 = 0
    count_5 = 0
    total = 0

    for idx, (query_img, query_idx, gt_label) in tqdm(enumerate(test_query_loader)):
        dis_label = dict()

        top1_distance_list = list()
        top1_distance_index = list()

        top5_distance_list = list()
        top5_distance_index = list()

        with torch.inference_mode():
            for i, (ref_img, ref_label) in enumerate(test_ref_loader):
                if ref_label.item() == gt_label.item(): 
                    label = 0
                else:
                    label = 1

                if type(loss_fn) == nn.TripletMarginLoss:
                    output1, output2 = model.forward_once(query_img), model.forward_once(ref_img)
                else:
                    # Contrastive Loss
                    output1, output2 = model(query_img, ref_img)
                    
                euclidean_distance = F.pairwise_distance(output1, output2).item()
                dis_label[euclidean_distance] = ref_label.item()

                if i % 500 == 0:
                    del ref_img, ref_label, output1, output2, euclidean_distance
                    gc.collect()
                    torch.cuda.empty_cache()
                    
        sorted_dislabel = sorted(dis_label.items(),key=lambda x:x[0])

        for (i, j) in sorted_dislabel[:ONE_PCNT]:
            top1_distance_list.append(i)
            top1_distance_index.append(j)
        
        for (i, j) in sorted_dislabel[:FIVE_PCNT]:
            top5_distance_list.append(i)
            top5_distance_index.append(j)

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
        
        if idx % 25 == 0:
            del gt_label
            gc.collect()
            torch.cuda.empty_cache()