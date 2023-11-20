import pandas as pd
import random
import os

from tqdm import tqdm
from PIL import Image
from pytorchtools import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms
import torchvision.models as models
import gc

from torchsummary import summary as Summary
import logging

torch.manual_seed(1000)

if torch.cuda.is_available():
    device='cuda:0'
    print('현재 가상환경 GPU 사용 가능상태')
else:
    device='cpu'
    print('GPU 사용 불가능 상태')


class Config:
    def __init__(self, loss_function, working_dir=""):
        self.loss_function = loss_function

        self.query_training_dir = os.path.join(working_dir, "query/train")
        self.query_test_dir = os.path.join(working_dir, "query/test")

        self.ref_trainig_dir = os.path.join(working_dir, "reference_v1")
        self.ref_test_dir = os.path.join(working_dir, "ref")

        self.image_labels = os.path.join(working_dir, "label_table.csv")
        
        self.train_val_split = [0.8, 0.2]
        
    
    def get_all_file_dirs(self):
        return {'query_training_dir': self.query_training_dir,
                'query_test_dir': self.query_test_dir,
                'ref_trainig_dir': self.ref_trainig_dir,
                'ref_test_dir': self.ref_test_dir,
                'image_labels': self.image_labels}
        
    
    def get_model_hyperparameters(self, option):
        if option == 1:
            return {"end_layer": 7,
                    "embedding_dim": 500,
                    "margin": 5.0,
                    "learning_rate": 0.0001,
                    "step_size": 5,
                    "gamma": 0.5}
    

EPOCHS = 300
BATCH_SIZE = 128
triplet_config = Config(loss_function="triplet", working_dir="test_feat_extractor")

class TrainingDataset(Dataset):
    def __init__(self, config):
        self.query_dir = config.query_training_dir
        self.ref_dir = config.ref_trainig_dir
        self.query_data_list = os.listdir(self.query_dir)
        self.ref_data_list = os.listdir(self.ref_dir)
        self.labeltable = pd.read_csv(config.image_labels)
        self.transform = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                             transforms.Resize((100,100)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.query_data_list)

    def __getitem__(self, idx):
        query = self.query_data_list[idx]
        query_number = int(query.split('.')[0])
        query_img = Image.open(os.path.join(self.query_dir, query))
        gt_query_label = int(self.labeltable[self.labeltable.cropped == query_number]['gt'].item())
        
        ref_version = random.choice([-1, 0, 1])
        if ref_version == -1:
            positive_ref_img = Image.open(os.path.join(self.ref_dir, \
                '{0:05d}'.format(gt_query_label) +'.png'))
        else:
            positive_ref_img = Image.open(os.path.join(self.ref_dir, \
                '{0:05d}'.format(gt_query_label) + '_' + str(ref_version) + '.jpg'))
        ref_label = gt_query_label

        while True:
            ref = random.choice(self.ref_data_list)
            negative_ref_label = int(ref.split('.')[0].split('_')[0])
            if gt_query_label != negative_ref_label:
                negative_ref_img = Image.open(os.path.join(self.ref_dir, ref))
                break

        query_img = self.transform(query_img).to(device)
        positive_ref_img = self.transform(positive_ref_img).to(device)
        negative_ref_img = self.transform(negative_ref_img).to(device)
        
        return query_img, positive_ref_img, negative_ref_img


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


train_valid_dataset = TrainingDataset(triplet_config)
train_dataset, valid_dataset = random_split(train_valid_dataset, triplet_config.train_val_split)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

cur_hyperparameters = triplet_config.get_model_hyperparameters(1)

resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
embedding_network = nn.Sequential(*list(resnet.children())[0:7]).to(device)

model = SiameseNetwork(embedding_network).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = nn.TripletMarginLoss(margin=5.0)

Summary(SiameseNetwork(embedding_network).to(device),[(3,100,100),(3,100,100), (3,100,100)])
logging.basicConfig(filename=f'{__file__}.log', \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')


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

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        
        return output1, output2, output3


resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
embedding_network = nn.Sequential(*list(resnet.children())[0:7]).to(device)

model = SiameseNetwork(embedding_network).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = nn.TripletMarginLoss(margin=5.0)

Summary(SiameseNetwork(embedding_network).to(device),[(3,100,100),(3,100,100), (3,100,100)])
logging.basicConfig(filename=f'{__file__}.log', \
        level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

save = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            # print(type(obj), obj.size(), end=' ')
            # print(obj.size(), end=' ')
            save.append(obj.size())
    except:
        pass
print(np.unique(save, return_counts=True))


def train_epoch(model, dataloader, loss_fn, optimizer):
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


def valid_epoch(model, dataloader, loss_fn):
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

BATCH_SIZE_TEST = 1
test_query_loader = DataLoader(test_query_dataset, batch_size=1, shuffle=False)
test_ref_loader= DataLoader(test_ref_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

model.eval()
labeltable = pd.read_csv(Config().image_labels)
count_1 = 0
count_5 = 0
total = 0

ONE_PCNT = int(len(test_ref_loader) * 0.01)
FIVE_PCNT = int(len(test_ref_loader) * 0.05)

counter = []
train_loss_history = []
valid_loss_history = []
iteration_number= 0
early_stopping = EarlyStopping(patience = 15, verbose = True)

for epoch in range(1,EPOCHS + 1):
    scheduler.step()
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer)
    valid_loss = valid_epoch(model, valid_dataloader, loss_fn)

    logging.info('Epoch number [{}] Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
    iteration_number += 1
    counter.append(iteration_number)
    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    early_stopping(valid_loss, model)
    
    if early_stopping.early_stop:
        break


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