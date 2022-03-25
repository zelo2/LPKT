# coding: utf-8
# started on 2022/3/22 @zelo2
# finished on 2022/?/? @zelo2

import torch
import numpy as np
import tqdm
from assist_chall import assistchall_process
import pandas as pd
from Freyja_LPKT import LPKTNet_copy
from Freyja_LPKT import LPKT_dataloader
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics


def train(dataset):
    if dataset == 'assist_chall':
        path = '../assist_chall/assist_chall_4LPKT.csv'
    elif dataset == 'assist_2012':
        path = '../assist_2012/assits_2012_4LPKT.csv'

    data_inf, data_sum = assistchall_process.data_split(pd.read_csv(path, encoding="utf-8", low_memory=True))


    '''Paramater Initialization'''
    stu_num = data_inf[0]
    exercise_num = data_inf[1]
    skill_num = data_inf[2]
    answer_time_num = data_inf[3]
    interval_time_num = data_inf[4]

    q_matrix = data_sum[0]
    raw_data = data_sum[1]  # [problem_id, answer time, interval time, correct]

    device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')

    acc = []
    auc = []


    '''5-fold cross validation'''
    n_split = 5
    kf = KFold(n_splits=n_split, shuffle=True)
    kf_train_vali = KFold(n_splits=n_split, shuffle=True)
    train = []
    vali = []
    test = []

    for train_plus_vali_index, test_index in kf.split(np.arange(len(raw_data))):
        for train_index, vali_index in kf_train_vali.split(train_plus_vali_index):
            test.append(raw_data[test_index])
            train.append(raw_data[train_index])
            vali.append(raw_data[vali_index])



    for fold in range(len(train_index)):

        '''Dataset'''
        train_dataset = LPKT_dataloader.lpkt_dataset(train[fold])
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        vali_dataset = LPKT_dataloader.lpkt_dataset(vali[fold])
        vali_dataloader = DataLoader(vali_dataset, batch_size=32, shuffle=True)

        test_dataset = LPKT_dataloader.lpkt_dataset(test[fold])
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        '''Inilization'''
        net = LPKTNet_copy.LPKTNet(exercise_num, skill_num, stu_num, answer_time_num, interval_time_num,
                                   d_k=128, d_a=50, d_e=128, q_matrix=q_matrix)
        net = net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

        '''Train and Validation'''
        for epoch in range(50):
            running_loss = 0
            print('Epoch', epoch+1)
            '''Train'''
            for _, (input_data, labels) in enumerate(train_dataloader):

                optimizer.zero_grad()
                labels = labels.float().to(device)

                input_data = input_data.to(device)
                exercise_id = input_data[:, 0].long()  # [batch_size, sequence]
                answer_time = input_data[:, 1].long()
                interval_time = input_data[:, 2].long()
                answer_value = input_data[:, 3].float()

                pred = net(exercise_id, answer_time, interval_time, answer_value)
                pred = pred[:, 1:].to(device)

                '''Backward propagation'''
                loss = loss_function(pred, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('Loss:', running_loss)

            '''Validation'''
            vali_pred = []
            vali_labels = []
            with torch.no_grad():
                for _, (input_data, labels) in enumerate(vali_dataloader):

                    labels = labels[:, -1].float().to(device)

                    input_data = input_data.to(device)  # [batch_size, 4, sequence]
                    exercise_id = input_data[:, 0].long()  # [batch_size, sequence]
                    answer_time = input_data[:, 1].long()
                    interval_time = input_data[:, 2].long()
                    answer_value = input_data[:, 3].float()

                    pred = net(exercise_id, answer_time, interval_time, answer_value)
                    pred = pred[:, -1].to(device)

                    vali_pred.append(pred)
                    vali_labels.append(labels)

                vali_pred = torch.cat(vali_pred).numpy().cpu()
                vali_labels = torch.cat(vali_labels).numpy().cpu()


            '''AUC'''
            vali_auc = metrics.roc_auc_score(vali_labels, vali_pred)

            '''ACC'''
            vali_pred[vali_pred >= 0.5] = 1
            vali_pred[vali_pred < 0.5] = 0
            vali_acc = np.nanmean((vali_pred == vali_labels) * 1)

            print("Validation of epoch (AUC, Acc)", epoch, ":", vali_auc, vali_acc)

        '''Test'''
        test_pred = []
        test_labels = []
        with torch.no_grad():
            for _, (input_data, labels) in enumerate(test_dataloader):
                labels = labels[:, -1].float().to(device)

                input_data = input_data.to(device)
                exercise_id = input_data[:, 0].long()  # [batch_size, sequence]
                answer_time = input_data[:, 1].long()
                interval_time = input_data[:, 2].long()
                answer_value = input_data[:, 3].float()

                pred = net(exercise_id, answer_time, interval_time, answer_value)
                pred = pred[:, -1].to(device)

                test_pred.append(pred)
                test_labels.append(labels)

            test_pred = torch.cat(test_pred).numpy().cpu()
            test_labels = torch.cat(test_labels).numpy().cpu()

            '''AUC'''
            test_auc = metrics.roc_auc_score(test_labels, test_pred)

            '''ACC'''
            test_pred[test_pred >= 0.5] = 1
            test_pred[test_pred < 0.5] = 0
            test_acc = np.nanmean((test_pred == test_labels) * 1)

            acc.append(test_acc)
            auc.append(test_auc)

            print("Test Results (AUC, Acc)", epoch, ":", test_auc, test_acc)

    print("Final Results (AUC, Acc):", np.mean(np.array(auc)), np.mean(np.array(acc)))

if __name__ == '__main__':
    dataset = ['assist_chall', 'assist_2012', 'Ednet']
    train(dataset[0])
