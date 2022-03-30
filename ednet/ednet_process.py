# started on 2022/3/30
# finished on 2022/?/? @zelo2

import os
import pandas as pd
import numpy as np
import tqdm
import torch

'''
Ednet_KT1:
timestamp: Unix timestamp in milliseconds
solving_id
question_id
user_answer: E.g., 'c'
elapsed_time: answer time (milliseconds)
'''


def id_dic_construction(x):
    corresponding_dic = {}
    for dic_index in range(len(x)):
        corresponding_dic[x[dic_index]] = dic_index + 1  # plus 1 for zero padding
    return corresponding_dic


def zero_padding(raw_kt_object, threshold=100):
    kt_object = []
    for student in tqdm.tqdm(range(len(raw_kt_object))):
        stu_object = raw_kt_object[student]

        stu_object_dim = stu_object.shape[0]
        stu_object_length = stu_object.shape[1]

        while stu_object_length > threshold:  # cut long sequence
            kt_object.append(stu_object[:, :threshold])
            stu_object = np.copy(stu_object[:, threshold:])
            stu_object_length -= threshold

        if stu_object_length > 0:  # complement short sequence
            complement_length = threshold - stu_object_length
            if complement_length == 0:  # exactly equal to sequence length
                kt_object.append(stu_object)
            else:
                stu_object = np.concatenate((np.zeros([stu_object_dim, complement_length]), stu_object), axis=1)
                kt_object.append(stu_object)

    return kt_object


def ednet_kt1_clean(question_path, data_path, threshold=15):
    '''
    Process the data as
    an uniform data format-['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct']
    '''

    ques_info = pd.read_csv(question_path, encoding="utf-8", low_memory=True)
    print(ques_info.columns)
    ques_info = ques_info.drop_duplicates()
    ques_info = ques_info.values

    '''delete "skill=-1" and its corresponding question'''
    delete_index = []
    delete_ques = []
    for i in range(len(ques_info)):
        skill = np.array(ques_info[i, -2].split(';')).astype('int64')  # tags
        if skill[0] == -1:
            delete_index.append(i)
            delete_ques.append(ques_info[i, 0])

    '''Construct "question" (exercise) and "skill" (knowledge concept) dictionaries'''
    ques_info = np.delete(ques_info, delete_index, axis=0)  # delete row
    ques_dic = id_dic_construction(ques_info[:, 0])  # question dictionary
    skill = []
    # ques_info_single_skill = []
    for i in range(len(ques_info)):
        ques_info[i, 0] = ques_dic[ques_info[i, 0]]
        skill += ques_info[i, -2].split(';')  # tags
        # for kc in np.array(ques_info[i, -2].split(';')).astype('int64'):
        #     ques_info_single_skill.append([ques_info[i, 0], kc, ques_info[i, 3]])
    skill = np.unique(np.array(skill).astype('int64'))
    skill_dic = id_dic_construction(skill)  # skill dictionary

    # '''Decode skill'''
    # for i in range(len(ques_info_single_skill)):
    #     ques_info_single_skill[i][1] = skill_dic[ques_info_single_skill[i][1]]
    # ques_info_single_skill = np.array(ques_info_single_skill)  # question contains multiple skills would be divided into multiple rows

    '''Process student record'''
    kt_data = []
    stu_file_list = os.listdir(data_path)
    # ['timestamp', 'solving_id', 'question_id', 'user_answer', 'elapsed_time']
    stu_id = 0
    for stu_file_name in tqdm.tqdm(stu_file_list):
        stu_record = pd.read_csv(data_path + '\\' + stu_file_name, encoding="utf-8", low_memory=True)
        stu_record = stu_record.drop_duplicates()

        # numpy array
        stu_record = stu_record.values

        # filter student
        delete_index_2 = []
        for i in range(len(stu_record)):
            if stu_record[i, 2] in delete_ques:
                delete_index_2.append(i)
        stu_record = np.delete(stu_record, delete_index_2, axis=0)  # row delete

        stu_length = stu_record.shape[0]
        if stu_length < threshold:
            continue

        lpkt_cell = []  # [? * 6]('user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct')
        for i in range(len(stu_record)):
            temp_lpkt_cell = []
            ques_id = ques_dic[stu_record[i, 2]]
            skill_index = np.where(ques_info[:, 0] == ques_id)[0][0]
            skill = ques_info[skill_index, -2][0].split(';')

            if stu_record[i, 3] == ques_info[skill_index, 3]:
                correct = 1
            else:
                correct = 0
            for kc in np.array(skill).astype('int64'):
                temp_lpkt_cell.append([stu_id, ques_id, kc,
                                       stu_record[i, 0]/1000,
                                       (stu_record[i, 0]+stu_record[i, -1])/1000,
                                       correct])
        kt_data.append(torch.tensor(temp_lpkt_cell))
        stu_id += 1
    kt_data = torch.cat(kt_data).numpy()
    processed_data = pd.DataFrame(kt_data,
                                  columns=['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct'])
    processed_data.to_csv('ednet_kt1_4LPKT.csv', index=False)
    return processed_data


if __name__ == '__main__':
    data_path = "E:\PycharmProject\ednet\EdNet\KT1"  # original path
    ques_path = "E:\PycharmProject\ednet\EdNet\EdNet-Contents\contents\questions.csv"
    ednet_kt1_clean(ques_path, data_path)
