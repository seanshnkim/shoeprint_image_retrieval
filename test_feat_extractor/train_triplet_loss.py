import numpy as np
import pandas as pd
from tqdm import tqdm

from pytorchtools import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

import torchvision.models as models
import gc

from torchsummary import summary as Summary
import logging

from configs.config import Config
from datasets.dataset import TrainDataset, TestQueryDataset, TestRefDataset
from nets.siamese_net import SiameseNetwork

torch.manual_seed(1000)

if torch.cuda.is_available():
    device='cuda:0'
    print('현재 가상환경 GPU 사용 가능상태')
else:
    device='cpu'
    print('GPU 사용 불가능 상태')


triplet_config = Config(loss="triplet", working_dir="test_feat_extractor")
model_hps = triplet_config.get_model_hyperparameters()
train_hps = triplet_config.get_training_hyperparameters()

train_valid_dataset = TrainDataset(triplet_config)
train_dataset, valid_dataset = random_split(train_valid_dataset, triplet_config.train_val_split)
train_dataloader = DataLoader(train_dataset, batch_size=train_hps["batch_size"], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=train_hps["batch_size"], shuffle=True)

if model_hps.model_name == "resnet50":
    resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
else:
    resnet = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')

embedding_network = nn.Sequential(*list(resnet.children())[:model_hps["end_layer"]]).to(device)

model = SiameseNetwork(embedding_network).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=train_hps["learning_rate"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_hps["step_size"], gamma=train_hps["gamma"])
loss_fn = nn.TripletMarginLoss(margin=train_hps["margin"], p=2)

Summary(SiameseNetwork(embedding_network).to(device),[(3,100,100),(3,100,100), (3,100,100)])

logging.basicConfig(filename=f'{__file__}.log', \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')


save = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            save.append(obj.size())
    except:
        pass
print(np.unique(save, return_counts=True))


def train_per_epoch(model, dataloader, loss_fn, optimizer):
    train_loss = 0.0
    model.train()
    model.requires_grad_ = True
    model.zero_grad()
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, img2 = data
        img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)
        
        optimizer.zero_grad()
        output1, output2, output3 = model(img0, img1, img2)
        
        del img0, img1, img2
        if i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        loss = loss_fn(output1, output2, output3)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        del loss, output1, output2, output3
        if i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    train_loss /= len(dataloader.dataset)
    return train_loss


def valid_per_epoch(model, dataloader, loss_fn):
    valid_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            img0, img1, img2 = data
            img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)
            
            output1, output2, output3 = model(img0, img1, img2)
            loss = loss_fn(output1, output2, output3)
            valid_loss += loss.item()

            del loss, output1, output2, output3
            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    
    valid_loss /= len(dataloader.dataset)
    return valid_loss


test_query_dataset = TestQueryDataset(Config())
test_ref_dataset = TestRefDataset(Config())

test_query_loader = DataLoader(test_query_dataset, batch_size=1, shuffle=False)
BATCH_SIZE_TEST = 1
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

ONE_PCNT = int(len(test_ref_loader) * 0.01)
FIVE_PCNT = int(len(test_ref_loader) * 0.05)

counter = []
train_loss_history = []
valid_loss_history = []
iteration_number= 0
early_stopping = EarlyStopping(patience=train_hps.early_stopping, verbose = True)


for epoch in range(1, train_hps.epochs + 1):
    scheduler.step()
    train_loss = train_per_epoch(model, train_dataloader, loss_fn, optimizer)
    valid_loss = valid_per_epoch(model, valid_dataloader, loss_fn)

    logging.info('Epoch number [{}] Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
    iteration_number += 1
    counter.append(iteration_number)
    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    early_stopping(valid_loss, model)
    
    if early_stopping.early_stop:
        break


model.eval()
labeltable = pd.read_csv(Config().image_labels)
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
            
            # query_img.shape = torch.Size([1, 3, 100, 100]) / ref_img.shape = torch.Size([1, 3, 100, 100])
            output1, output2 = model(query_img, ref_img)
            euclidean_distance = F.pairwise_distance(output1, output2).item()

            dis_label[euclidean_distance] = ref_label.item()

            if i % 500 == 0:
                del ref_img, ref_label, output1, output2, euclidean_distance
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