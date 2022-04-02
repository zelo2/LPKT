# coding: utf-8
# started on 2022/3/24 @zelo2
# finished on 2022/3/27 @zelo2
import numpy as np
import pandas as pd
import tqdm
import random
import torch

'''
ASSIST 2012 Dataset:

student_id：学生id  
skill：知识点名称
problem_id：问题id
start_time：答题时间
end_time：答题结束时间
correct：Response to a problem（0 or 1）


PS: For more detailed information about this dataset, please refer to 
https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect
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

def time_transform(time_string):
    '''
    :param time_string: e.g. "2012-10-09 11:01:52"
    :return: np.array-[year, month, day, hour, minute, second]
    warning: time_string must be "list" type
    '''
    result = []
    for i in range(len(time_string)):
        time_1 = time_string[i].split('-')  # [year, month, day+time]
        time_2 = time_1[-1].split(' ')  # [day, hour+minute+seconds]
        time_3 = time_2[-1].split(':')  # [hour, minute, seconds]

        result.append([time_1[0], time_1[1], time_2[0], time_3[0], time_3[1], time_3[2]])

    return np.array(result).astype('float')


def time_computation(time_start, time_end):
    '''
    :param time_a: start_time
    :param time_b: end_time
    :return: interval/answer time(second)
    '''
    time_start = time_transform(time_start)
    time_end = time_transform(time_end)
    result = time_end - time_start

    minute = 60  # 60 seconds
    hour = minute * 60
    day = hour * 24
    month = day * 31
    year = day * 365

    time_table = np.array([year, month, day, hour, minute, 1])  # 1 stands for second
    result *= time_table
    result = np.sum(result, axis=1)  # sum each row
    result[result > month] = month

    return result


def data_clean(og_data, threshold_length=1):


    raw_data = og_data[['user_id', 'problem_id', 'skill_id', 'start_time', 'end_time', 'correct']]
    raw_data = raw_data.drop_duplicates()

    raw_data = np.array(raw_data)  # [stu_id, item_id, skill, start_time, end_time,  answer]


    temp_skill = raw_data[:, 2]
    delete_index = []
    for i in range(len(temp_skill)):
        if np.isnan(temp_skill[i]):
            delete_index.append(i)
    # delete_index = np.unique(delete_index)  # ?
    delete_index = np.array(delete_index)



    raw_data = np.delete(raw_data, delete_index, axis=0)  # delete NaN

    temp_skill_id = np.unique(raw_data[:, 2])
    skill_id = []
    for i in temp_skill_id:
        if not np.isnan(i):
            skill_id.append(i)
    skill_id = np.array(skill_id)
    skill_dic = id_dic_construction(skill_id)
    print("skill number:", len(skill_id))

    raw_stu_id = raw_data[:, 0]
    stu_id = np.unique(raw_stu_id)

    for i in range(len(raw_data)):
        raw_data[i, 2] = skill_dic[raw_data[i, 2]]


    for i in tqdm.tqdm(range(len(stu_id))):
        stu_object = []
        student = stu_id[i]
        for j in range(len(raw_data)):
            if student == raw_data[j, 0]:
                stu_object.append(raw_data[j])

        if len(stu_object) < threshold_length:
            delete_stu = np.where(raw_data[:, 0] == student)[0]
            raw_data = np.delete(raw_data, delete_stu, axis=0)  # delete length less than 15

    raw_data = pd.DataFrame(raw_data,
                            columns=['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct'])

    return raw_data



def data_process_4LPKT(raw_data):

    raw_data = raw_data[['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct']]
    raw_data = raw_data.drop_duplicates()

    raw_data = np.array(raw_data)  # [stu_id, item_id, skill, start_time, end_time, answer]


    raw_stu_id = raw_data[:, 0]
    raw_exercise_id = raw_data[:, 1]
    raw_skill = raw_data[:, 2]



    raw_stu_id = np.unique(raw_stu_id)
    raw_exercise_id = np.unique(raw_exercise_id)

    # Information of Dataset
    stu_num = len(raw_stu_id)
    exercise_num = len(raw_exercise_id)

    print("Student Number:", stu_num)
    print("Exercise Number:", exercise_num)


    stu_dic = id_dic_construction(raw_stu_id)
    exercise_dic = id_dic_construction(raw_exercise_id)

    for i in range(len(raw_data)):
        raw_data[i, 0] = stu_dic[raw_data[i, 0]]
        raw_data[i, 1] = exercise_dic[raw_data[i, 1]]
        

    processed_data = pd.DataFrame(raw_data,
                                  columns=['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct'])
    processed_data.to_csv('assist_2012_4LPKT.csv', index=False)
    return processed_data



def data_split(raw_data, percent=None):
    '''
    :param raw_data: ['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct']
    :param percent: Ratio of training data
    :return:
    '''

    raw_data = raw_data.sort_values(by=['start_time'])
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

        # ['user_id', 'problem_id', 'skill', 'start_time', 'end_time', 'correct']
        stu_object = np.array(stu_object)
        # stu_object = stu_object.T
        if len(stu_object) == 1:
            continue
        # Answer Time Process
        answer_time = time_computation(time_start=stu_object[:, 3], time_end=stu_object[:, 4])

        round_mark_1 = (answer_time >= 0.5) * 1
        round_mark_2 = (answer_time < 1) * 1
        round_mark = (round_mark_1 + round_mark_2) == 2

        answer_time[round_mark] = 1  # Python error: np.around(0.5) = 0
        answer_time = np.around(answer_time)

        # Interval Time Computation
        start_time = stu_object[:, 3]
        interval_time = np.zeros(len(start_time))
        interval_time[1:] = time_computation(time_start=start_time[:-1], time_end=start_time[1:])

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
    og_data = pd.read_csv("2012-2013-data-with-predictions-4-final.csv", encoding="utf-8", low_memory=True)
    clean_data = data_clean(og_data, threshold_length=1)
    LPKT_data = data_process_4LPKT(clean_data)
    LPKT_data = pd.read_csv("assist_2012_4LPKT.csv", encoding="utf-8", low_memory=True)
    data_information, data_sum = data_split(LPKT_data)
