import numpy as np
import tensorflow as tf
import copy
import random
from matplotlib import pyplot as plt
from env_qoe import env_qoe #环境
from Config import Config
import scipy.io as sio
from time import *

config = Config()
env = env_qoe(config)
group_of_x = config.group_of_x                 #决策变量x的组数（5个一组）
c_num = config.c_num
x_num = config.x_num
model = tf.saved_model.load("DQN_model_DD3L")
x_group = config.group_of_x
N = config.N
tile_num = config.tile_num
level_num = config.level_num

fov_num = config.fov_num
fov_1_num = config.fov_1_num
x_1_num = config.x_1_num
Dkc_num = config.Dkc_num
bf0_num = config.bf0_num
Q_num = config.QSP_num
S_num = config.QSP_num
P_num = config.QSP_num


# t = np.load('20.npy')
# sample_set = t.tolist()
# s1 = c + x
if_in = 0
###########读取训练集###########
# tra = np.load('QOE_C.npy')
# train_set = tra.tolist()
# train_set_len = len(train_set)
# # ########从训练集采样######
# sample_set = random.sample(train_set,20)
# m = np.array(sample_set)
# np.save('sample_set20', m)
#
# a = np.load('sample_set20.npy')
# a = np.load('QOE_C_one.npy')
# a = np.load('QOE_C_4.npy')

CY = 1
a = np.load('testset_python_high.npy')
sample_set = a.tolist()
if CY == 1:
    b = np.load('J_track_DDQN.npy')
    J_track = b.tolist()

######开始每组参数计算网络输出结果######
maxJ = np.empty((20))
minJ = np.empty((20))
randJ = np.empty((20))
model_J = []
model_X = np.empty((20,N,tile_num,level_num))
time_record = []
for i in range(20):
    begin_time = time()
    state_maxj = 0
    state1 = sample_set[i]
    # state1 = sample_set[0]
    c = state1[0:c_num]
    state = env.reset(c)  # 初始化环境，获得初始状态
    ##########
    state2 = c + x_group*[0,0,0,1]

    state2 = [state2, [1] * x_group]  # 已选择的x区域标记
    state_maxj = state2
    maxJ[i] = env.call(state_maxj)
    ##########
    state3 = c + x_group*[1,0,0,0]

    state3 = [state3, [1] * x_group]  # 已选择的x区域标记
    state_minj = state3
    minJ[i] = env.call(state_minj)
    ##########
    cc = 0
    ccc = x_group*[0,0,0,0]

    for jjj in range(group_of_x):
        action = random.randint(0,3)
        ccc[action+cc] = 1
        cc += 4
    state4 = c+ccc
    state4 = [state4, [1] * x_group]  # 已选择的x区域标记
    state_randj = state4
    randJ[i] = env.call(state_randj)
###############
# s1 = train_set[859]
# if s1 in train_set:
#     if_in = 1
# print(if_in)
# #################

# if sample_set[0] in train_set:
#     if_in = 1
# print(if_in)



###########给出模型输出##########
    for t in range(group_of_x):
        action = model.predict(np.expand_dims(state[0], axis=0)).numpy()  # 选择模型计算出的 Q Value 最大的动作
        action = action[0]
        s_temp = copy.deepcopy(state)
        move = env.move(s_temp, action)
        next_state = move[0]

        # print("第", t, "步:")
        # print(state)
        # print(action)
        # print(next_state)

        ss = copy.deepcopy(next_state[0])
        s = []
        for num in ss:
            s.append(float(num))
        state = copy.deepcopy(next_state)
    J = env.call(next_state)
    model_J.append(J)
    print("第", i, "组:")
    print(next_state[0][Q_num+fov_num+S_num+fov_1_num+P_num+x_1_num+bf0_num+
                          Dkc_num : Q_num+fov_num+S_num+fov_1_num+P_num+x_1_num+
                                         bf0_num+Dkc_num+x_num])
    X_temp = next_state[0][Q_num+fov_num+S_num+fov_1_num+P_num+x_1_num+bf0_num+
                          Dkc_num : Q_num+fov_num+S_num+fov_1_num+P_num+x_1_num+
                                         bf0_num+Dkc_num+x_num]

    count = 0
    for ii in range(N):
        for j in range(tile_num):
            for k in range(level_num):
                model_X[i][ii][j][k] = X_temp[count]
                count += 1

    end_time = time()
    time1 = end_time - begin_time
    time_record.append(time1)
    print(time1)
# print(model_J)

#########绘图########
sio.savemat('DQN_X_J.mat', {'model_J': model_J,
                            'model_X': model_X,
                            'random_J':randJ
                            })

plt.figure(1)
plt.plot(model_J,label = "model_J", color='red', linestyle='--',marker='|')
plt.plot(maxJ,label = "Max_J", color='blue', linestyle=':',marker='|')
plt.plot(minJ,label = "Min_J", color='orange', linestyle=':',marker='|')
plt.plot(randJ,label = "Rand_J", color='green', linestyle=':',marker='|')
plt.legend(loc=0,ncol=2)
plt.title('J comapre')
# plt.ylim(ymin=400, ymax=600)
plt.xlabel('Test set serial number')
plt.ylabel('value of J')
if CY == 1:
    plt.figure(2)
    # plt.title('测试某些函数')
    # plt.ylim(ymin=-2000, ymax=0)
    plt.xlabel('number of episode')
    plt.plot(J_track)
    # plt.plot(constrain)
    plt.ylabel('value of J')
plt.show()