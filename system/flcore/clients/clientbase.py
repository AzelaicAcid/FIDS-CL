# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from flcore.trainmodel.resnet import  resnet10, BPHSplit



class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args

        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.model = args.model
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        # self.set_model()
        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = self.set_optimizer()
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
    # def set_model(self):
    #     scale_factor = self.args.model_cd[self.id]
    #     width = [int(cur * scale_factor) for cur in [64, 128, 256, 512]]
    #     self.model = resnet10(features=width, num_classes=self.args.num_classes)
    #     proj = nn.Sequential(
    #         nn.Linear(width[-1], 512),
    #         nn.ReLU()
    #     )
    #     self.model.fc = nn.Identity()
    #     head = nn.Linear(512, self.args.num_classes)
    #     self.model = BPHSplit(self.model, proj, head).to(self.device)
    #     with open('model_arch.txt', 'a') as f:
    #         f.write(str(self.model))
    def set_optimizer(self):
        if self.args.optimizer == 'sdg':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Unsupported optimizer')

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            # 使用 tqdm 为数据加载循环添加进度条
            for x, y in tqdm(testloaderfull, desc="Testing", leave=False):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                # 获取预测值
                pred = torch.argmax(output, dim=1)

                # 收集真实标签和预测值
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())

        # 展平列表并转换为 numpy 数组
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)

        # 计算精度、召回率和 F1 分数
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                                   labels=np.arange(self.num_classes))

        # 汇总所有类别的结果
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
        f1_avg = 2 * (precision_avg * recall_avg) / (recall_avg + precision_avg)
        # self.cur_f1 = f1
        return accuracy, precision, recall, f1, precision_avg, recall_avg, f1_avg

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
    def set_loss_function_weight(self, w):
        w = torch.tensor(w).to(self.device).float()
        self.loss = nn.CrossEntropyLoss(weight=w).to(self.device)


        # for name, module in self.model.base.named_modules():
        #     if isinstance(module, nn.Conv2d):  # 检查是否为卷积层
        #         print(f"{name}: {module}")
        #     if isinstance(module, nn.Linear):
        #         print(f"{name}: {module}")
        # for name, module in self.model.head.named_modules():
        #     print(f"{name}: {module}")
