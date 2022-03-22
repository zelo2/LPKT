# coding: utf-8
# started on 2021/12/22 @zelo2
# finished on 2022/3/22 @zelo2

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LPKTNet(nn.Module):
    def __init__(self, exercise_num, skill_num, stu_num, ans_time_num, interval_time_num, d_k, d_a, d_e, q_matrix):
        '''
        :param exercise_num: 试题数量
        :param skill_num: 知识点数量
        :param stu_num: 学生数量
        :param ans_time_num: 做答时间数量
        :param interval_time_num: 相邻Learning cell之间的interval time的数量
        :param d_a: Dimension of the answer (0-All Zero, 1-All One)
        :param d_e: Dimension of the exercise
        :param d_k: Dimension of the skill
        '''
        super(LPKTNet, self).__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.exercise_num = exercise_num
        self.skill_num = skill_num
        self.stu_num = stu_num
        self.ans_time_num = ans_time_num
        self.interval_time_num = interval_time_num
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        '''Enhance q-matrix'''
        self.gamma = 0.03
        self.q_matrix = q_matrix
        self.q_matrix[self.q_matrix == 0] = self.gamma

        '''Dropout layer'''
        self.dropout = nn.Dropout(0.2)  # follow the original paper

        '''Exercise Embedding'''
        self.exercise_embed = nn.Embedding(self.exercise_num, self.d_e)


        '''Time Embedding'''
        self.ans_time_embed = nn.Embedding(self.ans_time_num, self.d_k)
        self.interval_time_embed = nn.Embedding(self.interval_time_num, self.d_k)


        '''MLP Construction'''
        # Learning gain Embedding
        # 这里的Embedding的input没有加入试题所考察的知识点向量 存疑？
        self.learning_embed_layer = nn.Linear(self.d_e + self.d_k + self.d_a, self.d_k)  # input = exercise + answer time+ answer
        torch.nn.init.xavier_normal(self.learning_embed_layer.weight)  # follow the original paper

        # Learning Obtain Layer
        self.learning_layer = nn.Linear(self.d_k * 4, self.d_k)  # input = l(t-1) + interval time + l(t) + h(t-1)
        torch.nn.init.xavier_normal(self.learning_layer.weight)

        # Learning Judge Layer
        self.learning_gate = nn.Linear(self.d_k * 4, self.d_k)  # input = l(t-1) + interval time + l(t) + h(t-1)
        torch.nn.init.xavier_normal(self.learning_gate.weight)

        # Forgetting Layer
        self.forgetting_gate = nn.Linear(self.d_k * 3, self.d_k)  # input = h(t-1) + learning gain (t) + interval time
        torch.nn.init.xavier_normal(self.forgetting_gate.weight)

        # Predicting Layer
        self.predicting_layer = nn.Linear(self.d_k * 2, self.d_k)  # input = exercise (t+1) + h (t)
        torch.nn.init.xavier_normal(self.predicting_layer.weight)


    def forward(self, exercise_id,  ans_time, interval_time, answer_value):
        '''
        :param exercise_id: 试题id序列  batch_size * sequence
        :param answer_value: 试题得分序列
        :param ans_time: 回答时间序列
        :param interval_time: 两次回答间隔时间序列 长度=前面的序列长度-1
        :return: Prediction
        E.g:
             exercise_id- 1, 2, 3, 4, 6
             answer_value- 1, 1, 0, 0, 0, 0
             ans_time- 5, 10, 15, 5, 20
             interval_time- 1000, 20000, 5000, 400
        '''

        batch_size, sequence_len = exercise_id.size(0), exercise_id.size(1)

        '''Supposing the units of the answer time and the interval time are both Second (s)'''
        # interval_time /= 60  # discretized by minutes
        # ans_time /= 1  # discretized by seconds

        '''Obtain the Embedding of each element'''
        exercise = self.exercise_embed(exercise_id)  # batch_size * sequence * d_e
        ans_time = self.ans_time_embed(ans_time)  # batch_size * sequence * d_k
        interval_time = self.interval_time_embed(interval_time)  # batch_size * sequence * d_k

        '''Preprocess the answer'''
        answer = answer_value.view(-1, 1)  # (batch_size * sequence) * 1
        answer = answer.repeat(1, self.d_a)  # (batch_size * sequence) * d_a
        answer = answer.view(batch_size, -1, self.d_a)  # batch_size * sequence * d_a

        '''Initial the learning embedding'''
        # 使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐
        # batch_size * sequence * (d_e + d_k + d_a)
        learning_emd = self.learning_embed_layer(torch.cat((exercise, ans_time, answer), 2))  # [batch_size, sequence, d_k]

        '''Past parameters'''
        h_tilde_pre = None  # h_t-1
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)
        h_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.skill_num, self.d_k))  # [knowledge, d_k]
        h_pre = h_pre.repeat(batch_size, 1, 1).to(device)  # [batch_size, knowledge, d_k]

        '''Batch size train'''
        # 每个作答序列，我们都需要两两拿出来进行训练
        for t in range(sequence_len - 1):
            learning_vector = learning_emd[:, t]  # batch_size * d_k


            ''' Initial the students' mastery
                假设a是一个tensor，那么把a看作最小单元：
                a.repeat(2)表示在复制1行2列a;
                a.repeat(3, 2)表示复制3行2列个a；
                a.repeat(3, 2, 1)表示复制3个2行1列个a。
            '''

            temp_exercise_id = exercise_id[:, t]  # batch_size
            knowledge_vector = self.q_matrix[temp_exercise_id].view(batch_size, 1, -1)  # [batch_size, 1, knowledge]


            '''bmm是两个三维张量相乘, 两个输入tensor维度是 (b×n×m)和 (b×m×p), 第一维b代表batch size，输出为(b×n×p)'''
            # [batch_size, 1, knowledge concept]x[batch_size, knowledge concept, dk)
            if h_tilde_pre is None:
                h_tilde_pre = knowledge_vector.bmm(h).view(batch_size, -1)  # [batch_size, dk)

            '''learning module'''
            # [batch_size, d_k]+[batch_size, d_k]+[batch_size, d_k]+[batch_size, d_k]->[batch_size, d_k]
            lg = self.learning_layer((torch.cat(learning_pre, interval_time[:, t+1], learning_vector, h_tilde_pre), 1))
            lg = self.tanh(lg)

            learning_gate_weight = self.learning_gate((torch.cat(learning_pre, interval_time[:, t+1], learning_vector, h_tilde_pre), 1))
            learning_gate_weight = self.sigmoid(learning_gate_weight)

            LG = learning_gate_weight * ((lg + 1) / 2).view(batch_size, 1, -1)  # [batch_size, 1, d_k]
            # [batch_size, knowledge, 1] [batch, 1, d_k]
            LG_tilde = self.dropout(knowledge_vector.transpose(1, 2).bmm(LG))  # [batch_size, knowledge, d_k]

            '''Forgetting module'''
            forget_factor = self.forgetting_gate(torch.cat((h_pre, LG, interval_time[:, t+1]), 1))
            forget_factor = self.sigmoid(forget_factor)

            '''Unpdating knwoledge state'''
            h = LG_tilde + forget_factor * h_pre

            '''Predicting module'''
            h_tilde = knowledge_vector.bmm(h).view(batch_size, -1)  # [batch_size, dk)
            prediction = self.sigmoid(self.predicting_layer((torch.cat(exercise[:, t+1])), 1))

            '''Updating past parameters'''
            h_pre = h
            h_tilde_pre = h_tilde
            learning_pre = learning_vector

        return prediction  # [batch_size, 1]


