

import time
from flcore.clients.clientFIDS_CL import clientFIDS_CL
from flcore.servers.serverbase import Server
from threading import Thread
import math
import random
import numpy as np


class FIDS_CL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients

        self.set_slow_clients()
        self.set_clients(clientAVG_W)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            for client in self.selected_clients:
                client.train()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            self.set_client_loss_function_weights(self.args.loss_score_start_epoch, self.args.loss_score_end_epoch)
            self.calculate_Fweight(self.args.f1_score_start_epoch, self.args.f1_score_end_epoch)

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.calculate_agg_weights(self.args.f1_score_start_epoch, self.args.f1_score_end_epoch)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        with open('time_cost.txt', 'a+') as f:
            f.write(
                f'EXPERIMENT_{self.args.algorithm}\t {sum(self.Budget[1:]) / len(self.Budget[1:])}'
            )
        max_f1 = max(self.rs_test_f1)
        max_f1_index = self.rs_test_f1.index(max_f1)
        print("Best Round:")
        print(f'accuracy: {self.rs_test_acc[max_f1_index]:.4f}')
        print(f'precision: {self.rs_test_pre[max_f1_index]:.4f}')
        print(f'recall: {self.rs_test_recall[max_f1_index]:.4f}')
        print(f'f1: {max_f1:.4f}')
        # with open(self.args.result_file, 'a+') as f:
        #     f.write(
        #         f'EXPERIMENT_{self.args.algorithm}_{self.args.dataset}_{self.args.goal}_num_clients{self.args.num_clients}\n')
        #     f.write(
        #         f'accuracy: {self.rs_test_acc[max_f1_index]:.4f}\t precision: {self.rs_test_pre[max_f1_index]:.4f}\t recall: {self.rs_test_recall[max_f1_index]:.4f}\t f1: {max_f1:.4f}\n')
        #     f.write(f'\n')

        self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG_W)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def calculate_Fweight(self, start_epoch, end_epoch):
        if len(self.rs_test_f1) < start_epoch or len(self.rs_test_f1) > end_epoch:
            return
        # 初始化用于计算每个类别所有客户端的平均F1分数的列表
        num_classes = len(self.clients[0].class_f1)
        class_f1_sums = np.zeros(num_classes)
        client_count = len(self.clients)

        # 遍历所有客户端，累计每个类的F1分数
        for client in self.clients:
            for i in range(num_classes):
                class_f1_sums[i] += client.class_f1[i]

        # 计算每个类的平均F1分数
        class_f1_avgs = class_f1_sums / client_count

        # 获取F1分数最低的N个类的索引
        lowest_f1_indices = np.argsort(class_f1_avgs)[:self.args.num_low_classes]

        # 为每个客户端计算加权F1分数
        for client in self.clients:
            lowest_class_f1_sum = sum([client.class_f1[i] for i in lowest_f1_indices])
            lowest_class_f1_avg = lowest_class_f1_sum / self.args.num_low_classes
            client_weighted_f1 = (1 - self.args.alpha) * np.exp(client.current_f1) + self.args.alpha * np.exp(
                lowest_class_f1_avg)
            client.weighted_f1 = client_weighted_f1

        # 打印选出的最低F1分数的类别
        print(f"Lowest {self.args.num_low_classes} F1 score classes: {lowest_f1_indices}")

    def calculate_agg_weights(self, start_epoch, end_epoch):
        if len(self.rs_test_f1) < start_epoch:
            # 如果当前轮次少于start_epoch，直接返回，不做任何处理
            return
        if len(self.rs_test_f1) > end_epoch:
            return
        uploaded_Fweights = [c.weighted_f1 for c in self.clients]
        uploaded_Fweights = [Fw / sum(uploaded_Fweights) for Fw in uploaded_Fweights]
        print('Normalized Fweights', uploaded_Fweights)
        self.uploaded_weights = [w * Fw for w, Fw in zip(self.uploaded_weights, uploaded_Fweights)]
        self.uploaded_weights = [w / sum(self.uploaded_weights) for w in self.uploaded_weights]
        print("agg weights:", self.uploaded_weights)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        # self.uploaded_Fweights = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                # self.uploaded_Fweights.append(client.weighted_f1)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        print('data volume weights', self.uploaded_weights)
        # print("Fweights:", self.uploaded_Fweights)
        # self.uploaded_Fweights = [Fw / sum(self.uploaded_Fweights) for Fw in self.uploaded_Fweights]
        # print("Normalized Fweights:", self.uploaded_Fweights)
        # self.uploaded_weights = [w * Fw for w, Fw in zip(self.uploaded_weights, self.uploaded_Fweights)]
        # self.uploaded_weights = [w / sum(self.uploaded_weights) for w in self.uploaded_weights]
        # print("Adjusted weights:", self.uploaded_weights)

    def calculate_client_Lweight(self, client, factor=1.0):
        # 初始化用于计算当前客户端每个类别的F1分数
        num_classes = self.num_classes

        # 获取客户端的F1分数
        class_f1 = client.class_f1

        # 计算该客户端每个类的损失权重
        # 根据当前客户端的F1分数，使用exp函数来增加差异性
        class_f1_weights = (1 / np.exp(class_f1 * factor)) / np.sum(1 / np.exp(class_f1 * factor))
        return class_f1_weights

    def set_client_loss_function_weights(self, start_epoch, end_epoch, factor=1.0):
        # 如果当前轮次少于start_epoch，直接返回，不做任何处理
        if len(self.rs_test_f1) < start_epoch:
            return
        # 如果当前轮次超过end_epoch，停止调整
        if len(self.rs_test_f1) > end_epoch:
            return

        # 遍历每个客户端，并为每个客户端单独设置其损失函数权重
        for client in self.clients:
            Lw = self.calculate_client_Lweight(client, factor)
            print(f"Setting loss weight for client {client.id}: {Lw}")
            client.set_loss_function_weight(Lw)
