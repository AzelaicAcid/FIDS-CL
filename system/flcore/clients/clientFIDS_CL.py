import numpy as np
import time
from flcore.clients.clientavg import clientAVG
from tqdm import tqdm
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


class clientFIDS_CL(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.current_f1 = 0.0
        self.class_f1 = []
        self.weighted_f1 = 1

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
        f1_avg = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
        self.class_f1 = f1
        self.current_f1 = f1_avg
        return accuracy, precision, recall, f1, precision_avg, recall_avg, f1_avg
