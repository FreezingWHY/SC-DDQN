import numpy as np
import random
import numpy as np
import pandas as pd
from Config import Config
import scipy.io as sio


class Training_set_import:
    def __init__(self, set_num, config):
        self.num = set_num
        self.c_num = config.c_num
        self.x_num = config.x_num
        self.N = config.N
        self.tile_num = config.tile_num  # 切块数量
        self.level_num = config.level_num  # 质量等级数量

    def get_set(self, save_or_load):
        #####保存#####
        if save_or_load == 1:

            s = []
            Q = np.empty((self.num, self.N, self.tile_num, self.level_num))
            P = np.empty((self.num, self.N, self.tile_num, self.level_num))
            P_synthesis = np.empty((self.num, self.N, self.tile_num, self.level_num))
            S = np.empty((self.num, self.N, self.tile_num, self.level_num))
            BW = np.empty((self.num, self.N))
            S_synthesis = np.empty((self.num, self.N, self.tile_num, self.level_num))
            fov = np.empty((self.num, self.N, self.tile_num))
            # fov = np.ones((self.num, self.N, self.tile_num))
            fov_1 = np.empty((self.num, 1, self.tile_num))
            x_1 = np.empty((self.num, self.tile_num, self.level_num))
            Dkc = np.empty((self.num, self.N, self.tile_num))
            bf0 = np.empty((self.num, 1))
            for num in range(self.num):
                ####随机Q生成器####

                for i in range(self.N):
                    for j in range(self.tile_num):
                        Q[num][i][j][0] = 96.5 + (108 - 96.5) * random.random()
                        Q[num][i][j][1] = Q[num][i][j][0] + 2.5 + (4 - 2.5) * random.random()
                        # Q[i][j][2] = Q[i][j][1] + 2.1+(3.1-2.1)*random.random()
                        Q[num][i][j][2] = Q[num][i][j][1] + 2.5 + (4 - 2.5) * random.random()
                        Q[num][i][j][3] = Q[num][i][j][2] + 2.5 + (4 - 2.5) * random.random()
                ####随机P生成器####

                for i in range(self.N):
                    for j in range(self.tile_num):
                        P[num][i][j][0] = round(2767 + (381589 - 2767) * random.random())
                        P[num][i][j][1] = P[num][i][j][0] * (1.2 + (1.23 - 1.2) * random.random())
                        # P[i][j][2] = P[i][j][1]*(1.235+(1.255-1.235)*random.random())
                        P[num][i][j][2] = P[num][i][j][1] * (1.22 + (1.285 - 1.22) * random.random())
                        P[num][i][j][3] = P[num][i][j][2] * (1.3 + (1.34 - 1.3) * random.random())
                ######由P计算的解码时间合成参数######

                for i in range(self.N):
                    for j in range(self.tile_num):
                        for k in range(self.level_num):
                            #P_synthesis[num][i][j][k] = (0.0002356 * P[num][i][j][k] + 13) / 6000
                            P_synthesis[num][i][j][k] = (0.0002356 * P[num][i][j][k] + 13) / 6000
                ####S生成器（根据P计算获得）####

                for i in range(self.N):
                    for j in range(self.tile_num):
                        S[num][i][j][0] = P[num][i][j][0] * 15 / 1024 / 1024 * 8 / 7
                        S[num][i][j][1] = P[num][i][j][1] * 15 / 1024 / 1024 * 8 / 7
                        # S[i][j][2] = P[i][j][2] * 15 / 1024 / 1024 * 8 / 7
                        S[num][i][j][2] = P[num][i][j][2] * 15 / 1024 / 1024 * 8 / 7
                        S[num][i][j][3] = P[num][i][j][3] * 15 / 1024 / 1024 * 8 / 7

                        # Smin = 2767 * 15 / 1024 / 1024 * 8 / 7
                        # Smax = 1014267 * 15 / 1024 / 1024 * 8 / 7
                ####随机带宽BW生成器####

                for i in range(self.N):
                    BW[num][i] = 100 + (400 - 100) * random.random()
                    # BW[num][i] = 200
                    # BW[num][i] = 50000 + (80000 - 50000) * random.random()
                ######由S和BW计算的下载时间合成参数######

                for i in range(self.N):
                    for j in range(self.tile_num):
                        for k in range(self.level_num):
                            S_synthesis[num][i][j][k] = S[num][i][j][k] / BW[num][i]
                            # S_synthesis[i][j][k] = S[i][j][k]
                #####随机FOV，FOV_1生成器#####

                for i in range(self.N):
                    seeed = random.random()
                    if seeed <= 0.7:
                        fov[num][i] = np.random.choice([0, 1], size=self.tile_num, p=[.1, .9])
                    elif seeed > 0.7 and seeed <= 0.8:
                        fov[num][i] = np.random.choice([0, 1], size=self.tile_num, p=[.2, .8])
                    elif seeed > 0.8 and seeed <= 0.9:
                        fov[num][i] = np.random.choice([0, 1], size=self.tile_num, p=[.3, .7])
                    else:
                        fov[num][i] = np.random.choice([0, 1], size=self.tile_num, p=[.4, .6])


                seeed = random.random()
                if seeed <= 0.7:
                    fov_1[num][0] = np.random.choice([0, 1], size=self.tile_num, p=[.1, .9])
                elif seeed > 0.7 and seeed <= 0.8:
                    fov_1[num][0] = np.random.choice([0, 1], size=self.tile_num, p=[.2, .8])
                elif seeed > 0.8 and seeed <= 0.9:
                    fov_1[num][0] = np.random.choice([0, 1], size=self.tile_num, p=[.3, .7])
                else:
                    fov_1[num][0] = np.random.choice([0, 1], size=self.tile_num, p=[.4, .6])

                #####随机x_1生成器#####

                for i in range(self.tile_num):
                    x_1[num][i] = self.level_num * [0]
                    zhizhen = random.randint(0, self.level_num - 1)
                    x_1[num][i][zhizhen] = 1
                #####随机Dkc生成器#####

                for i in range(self.N):
                    brand = random.random()
                    for j in range(self.tile_num):
                        Dkc[num][i][j] = (0.1 + (0.5 - 0.1) * random.random()) + (1.2 + (0.01 - 1.2) * brand)
                #####随机bf0生成器#####

                bf0[num][0] = 0.3333 + (0.7 - 0.3333) * random.random()
                if num % 6 == 0:
                    bf0[num][0] = 0.3333 + (0.34 - 0.3333) * random.random()
                # bf0[num][0] = 1 + (1.5 - 1) * random.random()
                # bf0[num][0] = 0.3333 + (0.45 - 0.3333) * random.random()
                #bf0[num][0] = 0.3333 + (0.34 - 0.3333) * random.random()
                # bf0[num][0] = 0.33333
                ##############将所有多维参数压成一维############
                Q_one = np.empty((self.N * self.tile_num * self.level_num))
                S_one = np.empty((self.N * self.tile_num * self.level_num))
                P_one = np.empty((self.N * self.tile_num * self.level_num))
                fov_one = np.empty((self.N * self.tile_num))
                x_1_one = np.empty((self.tile_num * self.level_num))
                Dkc_one = np.empty((self.N * self.tile_num))
                fov_1_one = np.empty((self.tile_num))

                zhizhen = 0
                for i in range(self.N):
                    for j in range(self.tile_num):
                        for k in range(self.level_num):
                            Q_one[zhizhen] = Q[num][i][j][k]
                            S_one[zhizhen] = S_synthesis[num][i][j][k]
                            P_one[zhizhen] = P_synthesis[num][i][j][k]
                            zhizhen += 1
                zhizhen = 0
                for i in range(self.N):
                    for j in range(self.tile_num):
                        fov_one[zhizhen] = fov[num][i][j]
                        Dkc_one[zhizhen] = Dkc[num][i][j]
                        zhizhen += 1
                zhizhen = 0
                for i in range(self.N):
                    for j in range(self.tile_num):
                        for k in range(self.level_num):
                            Q_one[zhizhen] = Q[num][i][j][k]
                            S_one[zhizhen] = S_synthesis[num][i][j][k]
                            P_one[zhizhen] = P_synthesis[num][i][j][k]
                            zhizhen += 1
                zhizhen = 0
                for j in range(self.tile_num):
                    for k in range(self.level_num):
                        x_1_one[zhizhen] = x_1[num][j][k]
                        zhizhen += 1
                zhizhen = 0
                for j in range(self.tile_num):
                    fov_1_one[zhizhen] = fov_1[num][0][j]
                    zhizhen += 1


                temp = list(Q_one) + list(fov_one) + list(S_one) + list(fov_1_one) + list(P_one) + list(x_1_one) + list(
                    bf0[num]) + list(Dkc_one)

                s.append(temp)

            m = np.array(s)
            # np.save('trainset_python_EXP_N=4', m)
            np.save('pipe_state', m)

            # ss = str(s)
            # f.write(ss)  # write 写入
            # f.write('\n')  # write 写入
            sio.savemat('trainset_matlab_EXP_N=5.mat',{'Q':Q,
                                              'S':S,
                                              'P':P,
                                              'BW': BW,
                                              'fov': fov,
                                              'fov_1': fov_1,
                                              'x_1': x_1,
                                              'Dkc': Dkc,
                                              'bf0': bf0,
                                              })


        #####读取#####
        elif save_or_load == 0:
            a = np.load('testset_python_lowestbf.npy')
            s = a.tolist()
            train_set_len = len(s)
            print(train_set_len)
        else:
            print("wrong input")
            s = "wrong"
        return s


config = Config()
set = Training_set_import(100, config)
a = set.get_set(1)

print(a)