import yaml
import numpy as np
from tqdm import tqdm
from time import strftime, localtime
import gc
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.models as models

import logging

from loss import ContrastiveLoss
from dataset import TestQueryDataset, TestRefDataset
from nets.siamese_net import SiameseNetwork
from utils import set_device

device = set_device()

# To reproduce nearly 100% identical results across runs, this code must be inserted.
SEED = 1000
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Choose loss function
loss_type = "contrastive" # contrastive or triplet
with open('test_feat_extractor/config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# choose which model ckpt to load
ckpt_path = os.path.join(cfg["working_dir"], "checkpoints", 'contrastive_11-27_1102.pt')

model_hps = cfg['model_hyperparameters']['option_0']
train_hps = cfg['training_hyperparameters']['option_0']

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
# FIVE_PCNT = int(len(test_ref_loader) * 0.05)
FIVE_PCNT = 50

    
if __name__ == "__main__":
    if loss_type== "triplet":
        loss_fn = nn.TripletMarginLoss(margin=train_hps["margin"], p=2)
    elif loss_type == "contrastive":
        loss_fn = ContrastiveLoss(margin=train_hps["margin"])
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
        
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    log_save_dir = os.path.join(cfg["working_dir"], 'logs', f'{loss_type}_test_{start_time_stamp}.log')
    logging.basicConfig(filename=log_save_dir, \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    # reproduce results. torch seed, numpy seed.
    logging.info(f"Device: {device}\n")
    logging.info(f"seed number: {SEED}")
    logging.info(f"Loss function: {loss_fn}\n")
    logging.info(f"Model Hyperparameters: {model_hps}\n")
    logging.info(f"Training Hyperparameters: {train_hps}\n")
    logging.info(f"Loaded model checkpoint: {ckpt_path}\n")
    
    if model_hps["model_name"] == "resnet50":
        resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
    else:
        resnet = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
        
    embedding_network = nn.Sequential(*list(resnet.children())[:model_hps["end_layer"]]).to(device)
    model = SiameseNetwork(embedding_network, end_layer=model_hps["end_layer"], embdim=model_hps["embedding_dim"]).to(device)
    
    model_ckpt = torch.load(ckpt_path)
    model.load_state_dict(model_ckpt)
    
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
        logging.info(f"top 1% distance index: {top1_distance_index}")
        logging.info(f"top 50 distance index: {top5_distance_index}")
        logging.info(f'count top 1% accuracy: {count_1} / total: {total}')
        logging.info(f'count top 50 accuracy: {count_5} / total: {total}\n')
        
        if idx % 25 == 0:
            del gt_label
            gc.collect()
            torch.cuda.empty_cache()