# coding: utf-8
# started on 2022/3/19 @zelo2
# finished on 2022/3/23 @zelo2
import numpy as np
import pandas as pd
import tqdm
import random
import torch

'''
ASSIST challenge Dataset:

studentId：学生id  
skill：知识点名称
problemId：问题id
startTime：答题时间
endTime：答题结束时间
timeTaken：Answer Time （Seconds）
correct：Response to a problem（0 or 1）
AveCorrect：平均分


PS: For more detailed information about this dataset, please refer to 
https://docs.google.com/spreadsheets/d/1QVUStXiRerWbH1X0P11rJ5IsuU2Xutu60D1SjpmTMlk/edit#gid=0
'''


def id_dic_construction(x):
    corresponding_dic = {}
    for dic_index in range(len(x)):
        corresponding_dic[x[dic_index]] = dic_index + 1  # for zero padding
    return corresponding_dic


def skill_dic_construction(raw_skill):
    skill_dic = {}
    skill_list = []
    for skill_index in raw_skill:
        if skill_index not in skill_list:
            skill_list.append(skill_index)
    for skill_index in range(len(skill_list)):
        skill_dic[skill_list[skill_index]] = skill_index

    return skill_dic

def zero_padding(raw_kt_object, threshold=100):
    kt_object = []
    for student in tqdm.tqdm(range(len(raw_kt_object))):
        stu_object = raw_kt_object[student]

        stu_object_dim = stu_object.shape[0]
        stu_object_length = stu_object.shape[1]

        while stu_object_length > threshold:   # cut long sequence
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


def data_process_4LPKT(raw_data):
    raw_data = raw_data[['studentId', 'problemId', 'skill', 'startTime', 'endTime', 'timeTaken', 'correct']].fillna(
        999999)
    raw_data = raw_data.drop_duplicates()

    raw_data = np.array(raw_data)  # [stu_id, item_id, skill, start_time, end_time, answer_time, answer]
    raw_stu_id = raw_data[:, 0]
    raw_exercise_id = raw_data[:, 1]
    raw_skill = raw_data[:, 2]

    # print(np.where(raw_data == 999999))  #  no missing values

    raw_stu_id = np.unique(raw_stu_id)
    raw_exercise_id = np.unique(raw_exercise_id)

    stu_dic = id_dic_construction(raw_stu_id)
    exercise_dic = id_dic_construction(raw_exercise_id)
    skill_dic = skill_dic_construction(raw_skill)

    for i in range(len(raw_data)):
        raw_data[i, 0] = stu_dic[raw_data[i, 0]]
        raw_data[i, 1] = exercise_dic[raw_data[i, 1]]
        raw_data[i, 2] = skill_dic[raw_data[i, 2]]

    processed_data = pd.DataFrame(raw_data,
                                  columns=['studentId', 'problemId', 'skill', 'startTime', 'endTime', 'timeTaken',
                                           'correct'])
    processed_data.to_csv('assist_chall_4LPKT.csv', index=False)
    return processed_data


def data_split(raw_data, percent=None):
    '''
    :param raw_data: ['studentId', 'problemId', 'skill', 'startTime', 'endTime', 'timeTaken', 'correct']
    :param percent: Ratio of training data
    :return:
    '''

    raw_data = raw_data.sort_values(by=['startTime'])
    raw_data = np.array(raw_data)


    raw_stu_id = raw_data[:, 0]
    raw_exercise_id = raw_data[:, 1]
    raw_skill = raw_data[:, 2]

    stu_id = np.unique(raw_stu_id)
    exercise_id = np.unique(raw_exercise_id)
    skill_id = np.unique(raw_skill)

    # Information of Dataset
    stu_num = len(stu_id)
    exercise_num = len(exercise_id)
    skill_num = len(skill_id)
    print("Student Number:", stu_num)
    print("Exercise Number:", exercise_num)
    print("Skill Number:", skill_num)

    '''Initialization for q-matrix'''
    q_matrix = np.zeros([exercise_num+1, skill_num])
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
        start_time = stu_object[:, 3]
        interval_time = np.zeros(len(start_time))
        interval_time[1:] = start_time[1:] - start_time[:-1]

        interval_time /= 60
        one_month = 60 * 24 * 30
        interval_time[interval_time > one_month] = one_month  # set the interval time longer than one month as one month

        # problem_id, answer time, interval time, correct
        LPKT_cell = np.zeros([4, stu_object.shape[0]])
        LPKT_cell[0] = stu_object[:, 1]
        LPKT_cell[1] = answer_time
        LPKT_cell[2] = interval_time
        LPKT_cell[3] = stu_object[:, -1]
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
    kt_object = zero_padding(raw_kt_object, threshold=500)
    kt_object = np.array(kt_object)

    # raw_answer_time = np.array([])
    # raw_interval_time = np.array([])
    # for i in tqdm.tqdm(range(len(raw_kt_object))):
    #     raw_answer_time = np.concatenate((raw_answer_time, raw_kt_object[i][1]))
    #     raw_interval_time = np.concatenate((raw_interval_time, raw_kt_object[i][2]))
    #
    # raw_answer_time = np.unique(raw_answer_time)
    # raw_interval_time = np.unique(raw_interval_time)


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


# if __name__ == '__main__':
#     # og_data = pd.read_csv("anonymized_full_release_competition_dataset.csv", encoding="utf-8", low_memory=True)
#     # LPKT_data = data_process_4LPKT(og_data)
#     LPKT_data = pd.read_csv("assist_chall_4LPKT.csv", encoding="utf-8", low_memory=True)
#     data_information, data_sum = data_split(LPKT_data)
