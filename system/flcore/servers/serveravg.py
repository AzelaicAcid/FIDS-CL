

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import random
import numpy as np


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        # self.adjust_client_model()
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.acc_threshold = args.acc_thre

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

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
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
        #     f.write(f'EXPERIMENT_{self.args.algorithm}_{self.args.dataset}_{self.args.goal}_num_clients{self.args.num_clients}\n')
        #     f.write(f'accuracy: {self.rs_test_acc[max_f1_index]:.4f}\t precision: {self.rs_test_pre[max_f1_index]:.4f}\t recall: {self.rs_test_recall[max_f1_index]:.4f}\t f1: {max_f1:.4f}\n')
        #     f.write(f'\n')
        # self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def adjust_client_model(self):
        assert (len(self.clients) > 0)

        for scale_factor, client in zip(self.args.model_cd, self.clients):
            if scale_factor != 1:
                client.adjust_model(scale_factor)
