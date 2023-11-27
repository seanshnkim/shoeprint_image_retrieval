import yaml
import numpy as np
from tqdm import tqdm
from time import strftime, localtime
import gc
import os
import random
# import heapq

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
SEED = 2023
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DO NOT CHANGE BATCH SIZE OF QUERY
BATCH_SIZE_QUERY = 1
BATCH_SIZE_REF = 128

# Choose loss function
loss_type = "contrastive" # contrastive or triplet
with open('test_feat_extractor/config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# choose which model ckpt to load
ckpt_path = os.path.join(cfg["working_dir"], "checkpoints", 'contrastive_11-27_1302.pt')

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
test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE_QUERY, shuffle=True)
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_REF, shuffle=True)

#FIXME -  WRONG CALCULATION OF 1% and 5% of test_ref_loader 
# > len(test_ref_loader) is now equal to (entire ref dataset size / BATCH_SIZE_REF)
# ONE_PCNT = int(len(test_ref_loader) * 0.01)
# FIVE_PCNT = int(len(test_ref_loader) * 0.05)
ONE_PCNT = int(len(test_ref_dataset) * 0.01)
FIVE_PCNT = int(len(test_ref_dataset) * 0.05)
TOP_50 = 50

    
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
    logging.info(f"Device: {device}")
    logging.info(f"seed number: {SEED}")
    logging.info(f"batch size for query: {BATCH_SIZE_QUERY}")
    logging.info(f"batch size for reference: {BATCH_SIZE_REF}")
    logging.info(f"Loss function: {loss_fn}")
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

    if BATCH_SIZE_QUERY != 1:
        raise ValueError("BATCH_SIZE_QUERY must be 1")
    
    for qidx, (query_img, query_idx, gt_label) in tqdm(enumerate(test_query_loader)):
        dist_label_tuple = []
        query_img = query_img.to(device)

        with torch.inference_mode():
            for i, (ref_img, ref_label) in enumerate(test_ref_loader):
                ref_img = ref_img.to(device)
                
                if type(loss_fn) == nn.TripletMarginLoss:
                    output1, output2 = model.forward_once(query_img), model.forward_once(ref_img)
                else:
                    # Contrastive Loss
                    output1, output2 = model(query_img, ref_img)

                euclidean_distance = F.pairwise_distance(output1, output2)
                euclidean_distance = euclidean_distance.cpu().detach().numpy()
                
                '''#FIXME - Error that took 3 hours to find!! 
                # --> Limiting the size of heap to 50 initially does not guarantee
                # that the top 50 elements are the smallest 50 elements. '''       
                # for j in range(len(euclidean_distance)):
                #     if len(top50_list) < FIVE_PCNT:
                #         heapq.heappush(top50_list, (euclidean_distance[j], ref_label[j].item()) )
                #     else:
                #         heapq.heappushpop(top50_list, (euclidean_distance[j], ref_label[j].item()) )
                # for j in range(len(euclidean_distance)):
                #     heapq.heappush(hq, (euclidean_distance[j], ref_label[j].item()) )
                
                for j in range(len(euclidean_distance)):
                    dist_label_tuple.append((euclidean_distance[j], ref_label[j].item()) )
        
        tup_sorted = sorted(dist_label_tuple, key=lambda x: x[0])
        top50_list = tup_sorted[:TOP_50]
        top1pct_list = tup_sorted[:ONE_PCNT]
        
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