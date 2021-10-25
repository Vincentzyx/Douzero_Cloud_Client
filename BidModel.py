# -*- coding: utf-8 -*-
# Created by: Vincentzyx
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import time
import torch.nn.functional as F


def EnvToOnehot(cards):
    Env2IdxMap = {3:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9,13:10,14:11,17:12,20:13,30:14}
    cards = [Env2IdxMap[i] for i in cards]
    Onehot = torch.zeros((4,15))
    for i in range(0, 15):
        Onehot[:cards.count(i),i] = 1
    return Onehot


def RealToOnehot(cards):
    RealCard2EnvCard = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
                        '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
                        'K': 10, 'A': 11, '2': 12, 'X': 13, 'D': 14}
    cards = [RealCard2EnvCard[c] for c in cards]
    Onehot = torch.zeros((4,15))
    for i in range(0, 15):
        Onehot[:cards.count(i),i] = 1
    return Onehot


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 1 * 60
        self.conv1 = nn.Conv1d(1, 16, kernel_size=(3,), padding=1)  # 32 * 60
        self.dense1 = nn.Linear(1020, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(128, 1)

    def forward(self, xi):
        x = xi.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = x.flatten(1, 2)
        x = torch.cat((x, xi), 1)
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))
        x = F.leaky_relu(self.dense4(x))
        x = self.dense5(x)
        return x


UseGPU = False
device = torch.device('cuda:0')
net_bid = Net()
net_bid.eval()
net_farmer = Net()
net_farmer.eval()
if UseGPU:
    net_bid = net_bid.to(device)
    net_farmer = net_farmer.to(device)
if os.path.exists("./bid_weights.pkl"):
    if torch.cuda.is_available():
        net_bid.load_state_dict(torch.load('./bid_weights.pkl'))
    else:
        net_bid.load_state_dict(torch.load('./bid_weights.pkl', map_location=torch.device("cpu")))
if os.path.exists("./farmer_weights.pkl"):
    if torch.cuda.is_available():
        net_farmer.load_state_dict(torch.load('./farmer_weights.pkl'))
    else:
        net_farmer.load_state_dict(torch.load('./farmer_weights.pkl', map_location=torch.device("cpu")))


def predict(cards):
    x = RealToOnehot(cards)
    if UseGPU:
        x = x.to(device)
    x = torch.flatten(x)
    x = x.unsqueeze(0)
    score_bid = net_bid(x)
    score_farmer = net_farmer(x)
    return score_bid[0].item(), score_farmer[0].item()


def predict_env(cards):
    x = EnvToOnehot(cards)
    if UseGPU:
        x = x.to(device)
    x = torch.flatten(x)
    x = x.unsqueeze(0)
    score_bid = net_bid(x)
    score_farmer = net_farmer(x)
    return score_bid[0].item(), score_farmer[0].item()