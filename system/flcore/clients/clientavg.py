

import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from tqdm import tqdm
from torch import nn
from ..trainmodel.resnet import *


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 为训练过程添加 tqdm 进度条
        for epoch in range(max_local_epochs):
            epoch_desc = f"Epoch {epoch + 1}/{max_local_epochs}"
            # 在epoch中加入进度条
            for i, (x, y) in enumerate(tqdm(trainloader, desc=epoch_desc, leave=False)):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    
        # print("================================")
        # print(f'client id {self.id}')
        # print(self.model)
