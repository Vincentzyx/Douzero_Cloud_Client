"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)


class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(3,),
                               stride=(stride,), padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,),
                               stride=(1,), padding=1, bias=False)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=(1,), stride=(stride,), bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class GeneralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 80
        self.conv1 = nn.Conv1d(40, 80, kernel_size=(3,),
                               stride=(2,), padding=1, bias=False) #1*27*80
        self.lstm = nn.LSTM(54, 128, batch_first=True)
        self.bn1 = nn.BatchNorm1d(80)
        self.layer1 = self._make_layer(BasicBlock, 80, 2, stride=2)  # 1*14*80
        self.layer2 = self._make_layer(BasicBlock, 160, 2, stride=2)  # 1*7*160
        self.layer3 = self._make_layer(BasicBlock, 320, 2, stride=2)  # 1*4*320
        self.linear1 = nn.Linear(320 * BasicBlock.expansion * 4 + 15 * 4 + 128, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4_1 = nn.Linear(256, 128)
        self.linear5_1 = nn.Linear(128, 1)
        self.linear4_2 = nn.Linear(256, 128)
        self.linear5_2 = nn.Linear(128, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        h_moves = z[:, -32:]
        out = F.relu(self.bn1(self.conv1(z)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1,2)
        out = torch.cat([x,x,x,x,out], dim=-1)
        lstm_out, (h_n, _) = self.lstm(h_moves)
        lstm_out = lstm_out[:, -1, :]
        out = torch.cat((out, lstm_out), dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out1 = F.leaky_relu_(self.linear4_1(out))
        out1 = F.leaky_relu_(self.linear5_1(out1))
        out2 = F.leaky_relu_(self.linear4_2(out))
        out2 = F.leaky_relu_(self.linear5_2(out2))
        if return_value:
            return dict(values=(out1, out2))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action_wp = torch.randint(out1.shape[0], (1,))[0]
                action_adp = action_wp
            else:
                action_wp = torch.argmax(out1,dim=0)[0]
                action_adp = torch.argmax(out2, dim=0)[0]
            return dict(action=(action_wp, action_adp), max_value=(torch.max(out1), torch.max(out2)))


class BidModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 1 * 114
        self.conv1 = nn.Conv1d(1, 32, kernel_size=(3,)) # 32 * 112
        # pooling 32 * 56
        self.conv2 = nn.Conv1d(32, 64, kernel_size=(3,)) # 64 * 54
        # pooling 64 * 27
        self.conv3 = nn.Conv1d(64, 128, kernel_size=(3,)) # 128 * 25
        # pooling 128 * 12
        self.dense1 = nn.Linear(1536, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 256)
        self.dense4 = nn.Linear(256, 2)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = torch.max_pool1d(x, 2)
        x = F.leaky_relu(self.conv2(x))
        x = torch.max_pool1d(x, 2)
        x = F.leaky_relu(self.conv3(x))
        x = torch.max_pool1d(x, 2)
        x = x.flatten(1, 2)
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))
        x = torch.softmax(self.dense4(x), -1)
        if return_value:
            return dict(values=x)
        else:
            x = x.squeeze(0)
            if torch.isnan(x[0]) or torch.isnan(x[1]):
                action = np.random.choice(2, p=[0.5, 0.5])
                print("Bid模型出现NAN", x)
                return dict(action=action, max_value=torch.max(x))
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                # action = torch.randint(x.shape[-1], (1,))[0]
                # prob = x.cpu().numpy()
                action = np.random.choice(2, p=[0.5, 0.5])
            else:
                prob = x.cpu().numpy()
                action = np.random.choice(2, p=[prob[0], 1-prob[0]])
            return dict(action=action, max_value=torch.max(x))


# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel
model_dict_new = {}
model_dict_new['landlord'] = GeneralModel
model_dict_new['landlord_up'] = GeneralModel
model_dict_new['landlord_down'] = GeneralModel
model_dict_new['bidding'] = BidModel
model_dict_lstm = {}
model_dict_lstm['landlord'] = GeneralModel
model_dict_lstm['landlord_up'] = GeneralModel
model_dict_lstm['landlord_down'] = GeneralModel

class General_Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        # model = GeneralModel().to(torch.device(device))
        self.models['landlord'] = GeneralModel1().to(torch.device(device))
        self.models['landlord_up'] = GeneralModel1().to(torch.device(device))
        self.models['landlord_down'] = GeneralModel1().to(torch.device(device))
        self.models['bidding'] = BidModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags, debug)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['bidding'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()
        self.models['bidding'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models

class OldModel:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LandlordLstmModel().to(torch.device(device))
        self.models['landlord_up'] = FarmerLstmModel().to(torch.device(device))
        self.models['landlord_down'] = FarmerLstmModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        # model = GeneralModel().to(torch.device(device))
        self.models['landlord'] = GeneralModel().to(torch.device(device))
        self.models['landlord_up'] = GeneralModel().to(torch.device(device))
        self.models['landlord_down'] = GeneralModel().to(torch.device(device))
        self.models['bidding'] = BidModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags, debug)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['bidding'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()
        self.models['bidding'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models









