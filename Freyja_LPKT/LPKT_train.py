# coding: utf-8
# started on 2022/3/22 @zelo2
# finished on 2022/?/? @zelo2

import torch
import numpy as np
import tqdm
from data.assist_chall import assistchall_process
import pandas as pd
from Freyja_LPKT import LPKTNet_copy
from sklearn.model_selection import KFold

def train(dataset):
    if dataset == 'assist_chall':
        path = '../data/assist_chall/assist_chall_4LPKT.csv'

    data_inf, data_sum = assistchall_process.data_split(pd.read_csv(path, encoding="utf-8", low_memory=True))


    '''Paramater Initialization'''
    stu_num = data_inf[0]
    exercise_num = data_inf[1]
    skill_num = data_inf[2]
    answer_time_num = data_inf[3]
    interval_time_num = data_inf[4]

    q_matrix = data_sum[0]
    raw_data = data_sum[1]  # [problem_id, answer time, interval time, correct]

    net = LPKTNet_copy.LPKTNet(exercise_num, skill_num, stu_num, answer_time_num, interval_time_num,
                               d_k=128, d_a=50, d_e=128, q_matrix=q_matrix)

    '''Data split'''








if __name__ == '__main__':
    dataset = ['assist_chall', 'Ednet']
    train(dataset[0])