import yaml
import torch
from torch import optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from pytorchtools import EarlyStopping

import os
import gc
import logging
from time import strftime, localtime

from loss import ContrastiveLoss
from nets.siamese_net import SiameseNetwork
from dataset import TrainDataset
from utils import set_device

device = set_device()
torch.manual_seed(1000)

def train_per_epoch(model, dataloader, loss_fn, optimizer):
    train_loss = 0.0
    model.train()
    model.requires_grad_ = True
    model.zero_grad()
    
    for i, data in enumerate(dataloader, 0):
        imgs = [img.to(device) for img in data]
        
        optimizer.zero_grad()
        
        if type(loss_fn) == nn.TripletMarginLoss:
            anchor, pos, neg = imgs
            outputs = model(anchor, pos, neg)
            loss = loss_fn(*outputs)
        else:
            # Contrastive Loss
            pos, neg, label = imgs
            outputs = model(pos, neg)
            loss = loss_fn(*outputs, label)
        
        del imgs
        if i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        del loss, outputs
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
            imgs = [img.to(device) for img in data]
            
            if type(loss_fn) == nn.TripletMarginLoss:
                anchor, pos, neg = imgs
                outputs = model(anchor, pos, neg)
                loss = loss_fn(*outputs)
            else:
                # Contrastive Loss
                pos, neg, label = imgs
                outputs = model(pos, neg)
                loss = loss_fn(*outputs, label)
            
            valid_loss += loss.item()

            del loss, outputs
            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    valid_loss /= len(dataloader.dataset)
    return valid_loss


if __name__ == "__main__":
    # Choose loss function
    loss_type = "triplet" # contrastive or triplet
    
    with open('test_feat_extractor/config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    model_hps = cfg['model_hyperparameters']['option_0']
    train_hps = cfg['training_hyperparameters']['option_0']
    
    if model_hps["model_name"] == "resnet50":
        resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
    else:
        resnet = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')

    embedding_network = nn.Sequential(*list(resnet.children())[:model_hps["end_layer"]]).to(device)

    if loss_type == "triplet":
        loss_fn = nn.TripletMarginLoss(margin=train_hps["margin"], p=2)
    else:
        loss_fn = ContrastiveLoss(margin=train_hps["margin"])
    model = SiameseNetwork(embedding_network, end_layer=model_hps["end_layer"], embdim=model_hps["embedding_dim"]).to(device)
    
    optimizer = optim.Adam(params=model.parameters(), lr=train_hps["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_hps["step_size"], gamma=train_hps["gamma"])
    
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    log_save_dir = os.path.join(cfg.working_dir, 'logs', f'{loss_type}_train_{start_time_stamp}.log')
    logging.basicConfig(filename=log_save_dir, \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logging.info(f"Model Hyperparameters: {model_hps}\n")
    logging.info(f"Training Hyperparameters: {train_hps}\n")

    counter = []
    train_loss_history = []
    valid_loss_history = []
    iteration_number= 0
    ckpt_path = os.path.join(cfg.working_dir, "checkpoints", f'{loss_type}_{start_time_stamp}.pt')
    early_stopping = EarlyStopping(patience=train_hps["early_stopping"], path=ckpt_path, verbose=True)
    
    train_valid_dataset = TrainDataset(cfg)
    train_dataset, valid_dataset = random_split(train_valid_dataset, cfg.train_val_split)
    train_dataloader = DataLoader(train_dataset, batch_size=train_hps["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_hps["batch_size"], shuffle=True)


    # ========= TRAINING =========
    for epoch in range(1, train_hps["epochs"] + 1):
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