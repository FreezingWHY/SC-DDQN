import copy
import numpy as np
import random
class env_qoe():
    def __init__(self,config):         #问题规模设置
        self.N = config.N          #预测区间
        self.tile_num =config.tile_num   #切块数量
        self.level_num = config.level_num  #质量等级数量
        self.x_num = config.x_num    #x总数
        self.x_group = config.group_of_x        #x组数
        self.c_num = config.c_num     #可变参数数量
        self.f = 10
        self.fps = 30

        self.fov_num = config.fov_num
        self.fov_1_num = config.fov_1_num
        self.x_1_num = config.x_1_num
        self.Dkc_num = config.Dkc_num
        self.bf0_num = config.bf0_num
        self.Q_num = config.QSP_num
        self.S_num = config.QSP_num
        self.P_num = config.QSP_num
        self.BW_num = config.BW_num
    def call(self, state):                #目标函数值
        ######参数提取######        注：P，S为合成参数
        Q_temp = state[0][0:self.Q_num]
        fov_temp = state[0][self.Q_num : self.Q_num+self.fov_num]
        S_temp = state[0][self.Q_num+self.fov_num : self.Q_num+self.fov_num+self.S_num]
        fov_1_temp = state[0][self.Q_num+self.fov_num+self.S_num : self.Q_num+self.fov_num+self.S_num+self.fov_1_num]
        BW = state[0][self.Q_num+self.fov_num+self.S_num+self.fov_1_num
                          : self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num]
        x_1_temp = state[0][self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num
                            : self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num+self.x_1_num]
        bf0 = state[0][self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num+self.x_1_num]
        Dkc_temp = state[0][self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num+self.x_1_num+self.bf0_num
                            : self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num+self.x_1_num
                              +self.bf0_num+self.Dkc_num]
        x_temp = state[0][self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num+self.x_1_num+self.bf0_num+
                          self.Dkc_num : self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num+self.x_1_num+
                                         self.bf0_num+self.Dkc_num+self.x_num]

        # x_1_temp = np.zeros(self.tile_num*self.level_num)
        #bf0 = 0.4 #初始buffer
        #######反归一化#######
        # Qmin = 96.5
        # Qmax = 120.8
        # Symin = 0.000041
        # Symax = 0.01842
        # Pymin = 0.00227
        # Pymax = 0.04199
        # for i in range(self.N * self.tile_num * self.level_num):
        #     # Q_temp[i] = (Q_temp[i])*(Qmax-Qmin)+Qmin
        #     S_temp[i] = (S_temp[i])*(Symax-Symin)+Symin
        #     P_temp[i] = (P_temp[i])*(Pymax-Pymin)+Pymin
        #####将参数转化为矩阵形式######
        Q = np.empty((self.N,self.tile_num,self.level_num))
        count = 0
        for i in range(self.N):
            BW[i] = BW[i]*100
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    Q[i][j][k] = Q_temp[count]
                    count +=1

        fov = np.empty((self.N,self.tile_num))
        count = 0
        for i in range(self.N):
            for j in range(self.tile_num):
                fov[i][j] = fov_temp[count]
                count +=1

        S = np.empty((self.N,self.tile_num,self.level_num))
        count = 0
        for i in range(self.N):
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    S[i][j][k] = S_temp[count]
                    count +=1

        fov_1 = np.empty((self.tile_num))
        count = 0
        for j in range(self.tile_num):
            fov_1[j] = fov_1_temp[count]
            count +=1

        P = np.empty((self.N,self.tile_num,self.level_num))
        count = 0
        for i in range(self.N):
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    P[i][j][k] = S[i][j][k]*7/8*1024*1024/15
                    P[i][j][k] = (0.0002356 * P[i][j][k] + 13) / 6000
                    count +=1

        x_1 = np.empty((self.tile_num, self.level_num))
        count = 0
        for j in range(self.tile_num):
            for k in range(self.level_num):
                x_1[j][k] = x_1_temp[count]
                count += 1

        Dkc = np.empty((self.N,self.tile_num))
        count = 0
        for i in range(self.N):
            for j in range(self.tile_num):
                Dkc[i][j] = Dkc_temp[count]
                count +=1

        x = np.empty((self.N, self.tile_num, self.level_num))
        count = 0
        for i in range(self.N):
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    x[i][j][k] = x_temp[count]
                    count += 1
        ############目标函数表达式############
        ######定义权重,并根据x更新buffer######
        w1 = 1
        w2 = -3000
        w3 = -1

        bf = np.empty((self.N+1))
        bf[0] = bf0
        for i in range(self.N):
            bf[i+1] = self.buffer(S[i],bf[i],P[i],x[i],fov[i],BW[i])
        #print(bf)
        ###客观质量###
        A = 0
        for i in range(self.N):
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    A += Q[i][j][k] * x[i][j][k] * fov[i][j]
        ###暂停时间###
        B = 0
        for i in range(self.N):
            B_N_add = 0
            download_time = 0
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    download_time += (S[i][j][k]/BW[i]) * x[i][j][k] * fov[i][j]

            decode_time = 0
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    decode_time += P[i][j][k] * x[i][j][k] * fov[i][j]

            B += max(download_time + decode_time - bf[i], 0)
        ###质量切换###
        C = 0
        for i in range(self.N):
            for j in range(self.tile_num):
                for k in range(self.level_num):
                    if i == 0:
                        C += (k * (x[i][j][k] * fov[i][j] - x_1[j][k] * fov_1[j]))*min(fov[i][j],fov_1[j])*Dkc[i][j]
                    else:
                        C += (k * (x[i][j][k] * fov[i][j] - x[i-1][j][k] * fov[i-1][j]))\
                             * min(fov[i][j],fov[i-1][j])*Dkc[i][j]
        ###QOE###
        QOE = w1*A + w2*B + w3*C
        return QOE

    def buffer(self,S,bfk,P,x,fov,Bw):         #buffer迭代公式
        download_time = 0
        for j in range(self.tile_num):
            for k in range(self.level_num):
                download_time += S[j][k]/Bw * x[j][k] * fov[j]
        decode_time = 0
        for j in range(self.tile_num):
            for k in range(self.level_num):
                decode_time += P[j][k] * x[j][k] * fov[j]
        buf = max(bfk - download_time - decode_time,0) + self.f/self.fps
        return buf

    def wrong_choose(self,state,action): #判读是否在未选的区间进行选择
        xx = state[self.Q_num+self.fov_num+self.S_num+self.fov_1_num+self.BW_num+self.x_1_num+self.bf0_num+self.Dkc_num
                   : self.Q_num+self.fov_num+self.S_num+self.fov_1_num
                     +self.BW_num+self.x_1_num+self.bf0_num+self.Dkc_num+self.x_num]
        label = np.zeros(self.x_group)
        for iii in range(self.x_num):
            if xx[iii] == 1:
                label[iii//self.level_num] = 1
        # label = state[1]
        b = np.where(np.array(label) == 1)
        #print(b[0])
        if(action//self.level_num in b[0] and len(b)>=1):
            return 1
        else:
            return 0

    def reset(self,c):                  #状态初始化
        state = c
        state = state + [0]*self.x_num
        state = [state,[0]*self.x_group]               #已选择的x区域标记
        return state

    def move(self,state,action):              #按照智能体给出的策略进行行动，返回下一状态，收益等信息
        state_1 = copy.deepcopy(state)
        state[0][self.c_num+action] = 1
        state[1][action//self.level_num] = 1
        reward = self.reward(state,state_1)
        ####state后接一串已选择组数的标记序列####
        #state[0] = state[0][0:self.x_num*4] + state[1]
        ####################################
        move = [state,reward]
        return move

    def reward(self,state,state_1):
        # reward = pow((self.call(state) - self.call(state_1))-95,2)
        reward = self.call(state) - self.call(state_1)
        if reward != 0 :
            if reward-95 > 0:
                reward = pow(reward-95,2)
            else:
                reward = -pow(reward-95, 2)
        return reward

    def random_action(self,state):

        way = random.randint(0, 1)
        label = state[1]
        if way == 0:
        ##动作空间中选择随机动作(根据当前状态，从未被选的组中选择)
            rand_choose_1 = random.randint(0,self.level_num-1)
            b = np.where(np.array(label) == 0)
            rand_choose_2 = random.choice(b[0])
            action = self.level_num * rand_choose_2 + rand_choose_1

        elif way == 1:
        ##### 动作空间中选择随机动作(从所有组中随机选择)
            action = random.randint(0,self.x_num-1)
        #     # 动作空间中选择随机动作(从已经被选择的组中随机选择)
        #     rand_choose_1 = random.randint(0, 4)
        #     b = np.where(np.array(label) == 1)
        #     rand_choose_2 = random.choice(b[0])
        #     action = 5 * rand_choose_2 + rand_choose_1
        return action

    def fov_count(self,state):
        fov = state[0][self.Q_num : self.Q_num+self.fov_num]
        count = 0
        for i in fov:
            if i == 1:
                count +=1
        return count

# test = env_qoe()
# c = []
# for i in range(test.c_num):
#     c.append(i)
# state = test.reset(c)
# print(state)
# QOE = test.call(state)
# action = 3
# state = test.move(state,action)
# action = 6
# state = test.move(state[0],action)
# print(state)
# action = 12
# o = test.wrong_choose(state[0],action)
# print(o)
