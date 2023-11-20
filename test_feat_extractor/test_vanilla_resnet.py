import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm
import logging
import gc

from PIL import Image


class Config():
    query_test_dir = "query/test"
    ref_test_dir = "ref"
    image_labels = "label_table.csv"


class TestQueryDataset(Dataset):
    def __init__(self, Config):
        self.query_dir = Config.query_test_dir
        self.query_data_list = os.listdir(self.query_dir)
        self.labeltable = pd.read_csv(Config.image_labels)
        self.transform = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                             transforms.Resize((100,100)),
                                             transforms.ToTensor()])
        
    def __len__(self):
        return len(self.query_data_list)
    
    def __getitem__(self, idx):
        query = self.query_data_list[idx]
        query_idx = int(query.split('.')[0])
        gt_query_label = int(self.labeltable[self.labeltable.cropped == query_idx]['gt'].item())
        query_img = Image.open(os.path.join(self.query_dir, query))
        query_img = self.transform(query_img).to(device)
        
        return query_img, query_idx, gt_query_label


class TestRefDataset(Dataset):
    def __init__(self, Config):
        self.ref_dir = Config.ref_test_dir
        self.ref_data_list = os.listdir(self.ref_dir)
        self.transform = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                             transforms.Resize((100,100)),
                                             transforms.ToTensor()])
        
    def __len__(self):
        return len(self.ref_data_list)
    
    def __getitem__(self, idx):
        ref = self.ref_data_list[idx]
        ref_idx = int(ref.split('.')[0])
        ref_img = Image.open(os.path.join(self.ref_dir, ref))
        ref_img = self.transform(ref_img).to(device)

        return ref_img, ref_idx
    


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_network):
        super(SiameseNetwork, self).__init__()
        self.embedding_network = embedding_network
        self.fc1 = nn.Sequential(
            # nn.Linear(256*25*25, 500),
            nn.Linear(1024*7*7, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.embedding_network(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    
test_query_dataset = TestQueryDataset(Config())
test_ref_dataset = TestRefDataset(Config())

BATCH_SIZE_TEST = 1
test_query_loader = DataLoader(test_query_dataset, batch_size=1, shuffle=False)
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

labeltable = pd.read_csv(Config().image_labels)
count_1 = 0
count_5 = 0
total = 0

ONE_PCNT = int(len(test_ref_loader) * 0.01)
FIVE_PCNT = int(len(test_ref_loader) * 0.05)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device available: {device}")

resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
embedding_network = nn.Sequential(*list(resnet.children())[0:7]).to(device)
model = SiameseNetwork(embedding_network).to(device)

logging.basicConfig(filename=f'{__file__}.log', \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')


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