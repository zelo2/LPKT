# started on 2022/3/30
# finished on 2022/4/1 @zelo2

import os
import pandas as pd
import numpy as np
import tqdm
import torch
import random

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


def skill_dic_construction(x):
    corresponding_dic = {}
    for dic_index in range(len(x)):
        corresponding_dic[x[dic_index]] = dic_index
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


def ednet_kt1_clean(question_path, data_path, threshold=1):
    '''
    Process the data as
    an uniform data format-['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'time_taken', 'correct']
    '''

    ques_info = pd.read_csv(question_path, encoding="utf-8", low_memory=True)
    print(ques_info.columns)
    ques_info = ques_info.drop_duplicates()
    ques_info = ques_info.values

    '''delete "skill=-1" and its corresponding question'''
    delete_index = []
    delete_ques = []
    for i in range(len(ques_info)):
        ques_skill = np.array(ques_info[i, -2].split(';')).astype('int64')  # tags
        if ques_skill[0] == -1:
            delete_index.append(i)
            delete_ques.append(int(ques_info[i, 0][1:]))

    '''Construct "question" (exercise) and "skill" (knowledge concept) dictionaries'''
    ques_info = np.delete(ques_info, delete_index, axis=0)  # delete row
    skill = []
    for i in range(len(ques_info)):
        ques_info[i, 0] = int(ques_info[i, 0][1:])
        skill += ques_info[i, -2].split(';')  # tags

    print("Exercise Number:", len(np.unique(ques_info[:, 0])))

    skill = np.unique(np.array(skill).astype('int64'))
    skill_dic = skill_dic_construction(skill)  # skill dictionary
    print(skill_dic)
    print("Skill Number:", len(skill))

    '''Process student record'''
    kt_data = []
    stu_file_list = os.listdir(data_path)
    stu_file_list = stu_file_list
    # ['timestamp', 'solving_id', 'question_id', 'user_answer', 'elapsed_time']
    stu_id = 0

    # ['timestamp', 'solving_id', 'question_id', 'user_answer', 'elapsed_time']
    # answer_dic = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    for stu_file_name in tqdm.tqdm(stu_file_list):
        stu_record = pd.read_csv(data_path + '\\' + stu_file_name, encoding="utf-8", low_memory=True).fillna(-2)
        stu_record = stu_record.drop_duplicates()
        # numpy array
        stu_record = stu_record.values

        # filter student
        delete_index_2 = []
        for i in range(len(stu_record)):
            stu_record[i, 2] = int(stu_record[i, 2][1:])
            if (stu_record[i, 2] in delete_ques) or (stu_record[i, 3] == -2):
                delete_index_2.append(i)
        stu_record = np.delete(stu_record, delete_index_2, axis=0)  # row delete

        stu_length = stu_record.shape[0]
        if stu_length < threshold:
            print(stu_id)
            continue

        # [? * 7]('user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'time_taken', 'correct')
        temp_lpkt_cell = []
        for i in range(len(stu_record)):

            ques_id = stu_record[i, 2]
            skill_index = np.where(ques_info[:, 0] == ques_id)[0][0]

            skill = ques_info[skill_index, -2].split(';')

            if stu_record[i, 3] == -2:
                print(stu_record[i])

            if stu_record[i, 3] == ques_info[skill_index, 3]:
                correct = 1
            else:
                correct = 0
            for kc in np.array(skill).astype('int64'):
                temp_lpkt_cell.append([stu_id, ques_id, kc,
                                       stu_record[i, 0] / 1000,
                                       (stu_record[i, 0]) / 1000 + stu_record[i, -1] / 1000,
                                       stu_record[i, -1] / 1000,
                                       correct])

        kt_data.append(torch.tensor(temp_lpkt_cell))
        # print(torch.tensor(temp_lpkt_cell).size())
        stu_id += 1
    kt_data = torch.cat(kt_data).numpy()
    processed_data = pd.DataFrame(kt_data,
                                  columns=['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'time_taken',
                                           'correct'])
    processed_data.to_csv('ednet_kt1_4LPKT.csv', index=False)
    return processed_data


def data_split(raw_data, percent=None):
    '''
    :param raw_data: ['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'time_taken', 'correct']
    :param percent: Ratio of training data
    :return:
    '''

    raw_data = raw_data.sort_values(by=['start_time'])
    print(raw_data.sort_values(by=['problem_id']))

    raw_data = np.array(raw_data)

    raw_stu_id = raw_data[:, 0]
    raw_exercise_id = raw_data[:, 1]
    raw_skill = raw_data[:, 2]

    stu_id = np.unique(raw_stu_id)
    exercise_id = np.unique(raw_exercise_id)
    skill_id = np.unique(raw_skill)

    exercise_dic = id_dic_construction(exercise_id)
    skill_dic = skill_dic_construction(skill_id)  # skill dictionary

    # exercise & skill recoding to avoid out of bound
    for i in range(len(raw_data)):
        raw_data[i, 1] = exercise_dic[raw_data[i, 1]]
        raw_data[i, 2] = skill_dic[raw_data[i, 2]]
    raw_exercise_id = raw_data[:, 1]
    raw_skill = raw_data[:, 2]

    # Information of Dataset
    stu_num = len(stu_id)
    exercise_num = len(exercise_id)
    skill_num = len(skill_id)
    print("Student Number:", stu_num)
    print(exercise_id)
    print("Exercise Number:", exercise_num)
    print("Skill Number:", skill_num)

    '''Initialization for q-matrix'''
    q_matrix = np.zeros([exercise_num + 1, skill_num])
    for i in range(len(raw_data)):
        q_matrix[int(raw_exercise_id[i]), int(raw_skill[i])] = 1
    q_matrix = torch.from_numpy(q_matrix)

    raw_kt_object = []
    for i in tqdm.tqdm(range(len(stu_id))):
        stu_object = []
        student = stu_id[i]
        for j in range(len(raw_data)):
            if student == raw_data[j, 0]:
                stu_object.append(raw_data[j])

        # ['studentId', 'problemId', 'skill', 'startTime', 'endTime', 'timeTaken', 'correct']
        stu_object = np.array(stu_object)
        # stu_object = stu_object.T

        # Answer Time Process
        answer_time = np.around(stu_object[:, -2])

        round_mark_1 = answer_time >= 0.5 * 1
        round_mark_2 = answer_time < 1 * 1
        round_mark = (round_mark_1 + round_mark_2) == 2

        answer_time[round_mark] = 1  # Python error: np.around(0.5) = 0
        answer_time = np.around(answer_time)

        # Interval Time Computation
        start_time = np.copy(stu_object[:, 3])
        interval_time = np.zeros(len(start_time))
        interval_time[1:] = start_time[1:] - start_time[:-1]

        interval_time /= 60
        one_month = 60 * 24 * 30
        interval_time[interval_time > one_month] = one_month  # set the interval time longer than one month as one month

        # problem_id, answer time, interval time, correct
        LPKT_cell = np.zeros([4, stu_object.shape[0]])
        LPKT_cell[0] = np.copy(stu_object[:, 1])
        LPKT_cell[1] = answer_time
        LPKT_cell[2] = interval_time
        LPKT_cell[3] = np.copy(stu_object[:, -1])
        LPKT_cell = LPKT_cell.astype('int64')

        raw_kt_object.append(LPKT_cell)

    '''Decode time parameters'''
    raw_answer_time = np.array([])
    raw_interval_time = np.array([])
    for i in tqdm.tqdm(range(len(raw_kt_object))):
        raw_answer_time = np.concatenate((raw_answer_time, raw_kt_object[i][1]))
        raw_interval_time = np.concatenate((raw_interval_time, raw_kt_object[i][2]))

    raw_answer_time = np.unique(raw_answer_time)
    raw_interval_time = np.unique(raw_interval_time)
    answer_time_num = len(raw_answer_time)
    interval_time_num = len(raw_interval_time)

    answer_time_dic = id_dic_construction(raw_answer_time)
    interval_time_dic = id_dic_construction(raw_interval_time)

    for i in tqdm.tqdm(range(len(raw_kt_object))):
        for j in range(len(raw_kt_object[i][0])):
            raw_kt_object[i][1][j] = answer_time_dic[raw_kt_object[i][1][j]]
            raw_kt_object[i][2][j] = interval_time_dic[raw_kt_object[i][2][j]]

    '''Zero Padding'''
    kt_object = zero_padding(raw_kt_object, threshold=100)
    kt_object = np.array(kt_object)

    if percent is not None:
        # End for
        random.shuffle(kt_object)

        train_val_len = int(len(kt_object) * percent)
        train_len = int(train_val_len * 0.8)

        train_data = kt_object[:train_len]

        val_data = kt_object[train_len:train_val_len]

        test_data = kt_object[train_val_len:]

        return [stu_num, exercise_num, skill_num, answer_time_num, interval_time_num], [q_matrix, train_data, val_data,
                                                                                        test_data]

    else:

        return [stu_num, exercise_num, skill_num, answer_time_num, interval_time_num], [q_matrix, kt_object]


if __name__ == '__main__':
    data_path = "E:\PycharmProject\ednet\EdNet\KT1"  # original path
    ques_path = "E:\PycharmProject\ednet\EdNet\EdNet-Contents\contents\questions.csv"
    ednet_kt1_clean(ques_path, data_path)
    data = pd.read_csv('../ednet/ednet_kt1_4LPKT.csv', encoding='utf-8', low_memory=True)
    data_split(data)
