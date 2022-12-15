import copy
import tensorflow as tf
import numpy as np
import random
from collections import deque
from matplotlib import pyplot as plt
from env_qoe import env_qoe #环境
import datetime
from time import *
from Exp_pool import Exp_pool
from Config import Config
##########测试是否使用GPU########
# import tensorflow as tf
# gpu_out=tf.config.list_physical_devices('GPU')
# print(gpu_out)
#######QoE参数（c，x）######
config = Config()
N = config.N

tile_num = config.tile_num  # 切块数量
level_num = config.tile_num  # 质量等级数量
group_of_x = config.group_of_x  # 决策变量x的组数（5个一组）-
c_num = config.c_num
x_num = config.x_num
s_len = config.s_len
memory_size = config.memory_size
batch_size = config.batch_size
#######DQN参数#######
num_episodes = config.num_episodes             # 训练的总episode数量
num_exploration_episodes = config.num_exploration_episodes  # 探索过程所占的episode数量
# max_len_episode = 1500          # 每个episode的最大回合数
batch_size = config.batch_size                # 批次大小
learning_rate = config.learning_rate            # 学习率
gamma = config.gamma                      # 折扣因子
initial_epsilon = config.initial_epsilon            # 探索起始时的探索率
final_epsilon = config.final_epsilon           # 探索终止时的探索率
class QNetwork(tf.keras.Model):
    @tf.function
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop1 = tf.keras.layers.Dropout(0.05)
        self.bn1 = tf.keras.layers.LayerNormalization(axis=-1)
        # self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dense2 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop2 = tf.keras.layers.Dropout(0.05)
        self.bn2 = tf.keras.layers.LayerNormalization(axis=-1)

        self.dense3 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop3 = tf.keras.layers.Dropout(0.05)
        self.bn3 = tf.keras.layers.LayerNormalization(axis=-1)

        # self.dense4 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        # self.drop4 = tf.keras.layers.Dropout(0.05)
        # self.bn4 = tf.keras.layers.LayerNormalization(axis=-1)

        self.V_dence = tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.02))

        self.A_dence = tf.keras.layers.Dense(units=x_num, activation=tf.keras.layers.LeakyReLU(alpha=0.02))


    @tf.function
    def call(self, inputs):         #输出Q值
        ########两层全连接######
        FN = self.dense1(inputs)
        FN = self.drop1(FN)
        FN = self.bn1(FN)
        FN = self.dense2(FN)
        FN = self.drop2(FN)
        FN = self.bn2(FN)
        FN = self.dense3(FN)
        FN = self.drop3(FN)
        FN = self.bn3(FN)
        # FN = self.dense4(FN)
        # FN = self.drop4(FN)
        # FN = self.bn4(FN)
        #########状态价值V函数########
        svalue = self.V_dence(FN)
        ##########状态下的动作价值函数A########
        avalue = self.A_dence(FN)
        mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1,keepdims=True))(avalue)   #用Lambda层，计算avg(a)
        advantage = avalue - mean             #a - avg(a)

        output = svalue + advantage

        x = output
        return x

    # @tf.function
    # def realcall(self, inputs):         #输出Q值
    #     x = self.dense1(inputs)
    #     x = self.dense2(x)
    #     x = self.dense3(x)
    #     # x = self.dense4(x)
    #     x = self.dense10(x)
    #     return x

    @tf.function
    def posibility(self,input):
        x = self.call(input)
        x = tf.nn.softmax(x)
        return x

    @tf.function
    def predict(self, inputs):      #输出行动
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)

class Target_QNetwork(tf.keras.Model):
    @tf.function
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop1 = tf.keras.layers.Dropout(0.05)
        self.bn1 = tf.keras.layers.LayerNormalization(axis=-1)
        # self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dense2 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop2 = tf.keras.layers.Dropout(0.05)
        self.bn2 = tf.keras.layers.LayerNormalization(axis=-1)

        self.dense3 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop3 = tf.keras.layers.Dropout(0.05)
        self.bn3 = tf.keras.layers.LayerNormalization(axis=-1)

        # self.dense4 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        # self.drop4 = tf.keras.layers.Dropout(0.05)
        # self.bn4 = tf.keras.layers.LayerNormalization(axis=-1)

        self.V_dence = tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.02))

        self.A_dence = tf.keras.layers.Dense(units=x_num, activation=tf.keras.layers.LeakyReLU(alpha=0.02))

    @tf.function
    def call(self, inputs):  # 输出Q值
        ########两层全连接######
        FN = self.dense1(inputs)
        FN = self.drop1(FN)
        FN = self.bn1(FN)
        FN = self.dense2(FN)
        FN = self.drop2(FN)
        FN = self.bn2(FN)
        FN = self.dense3(FN)
        FN = self.drop3(FN)
        FN = self.bn3(FN)
        # FN = self.dense4(FN)
        # FN = self.drop4(FN)
        # FN = self.bn4(FN)
        #########状态价值V函数########
        svalue = self.V_dence(FN)
        ##########状态下的动作价值函数A########
        avalue = self.A_dence(FN)
        mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(avalue)  # 用Lambda层，计算avg(a)
        advantage = avalue - mean  # a - avg(a)

        output = svalue + advantage

        x = output
        return x

    # @tf.function
    # def realcall(self, inputs):         #输出Q值
    #     x = self.dense1(inputs)
    #     x = self.dense2(x)
    #     x = self.dense3(x)
    #     # x = self.dense4(x)
    #     x = self.dense10(x)
    #     return x

    def posibility(self,input):
        x = self.call(input)
        x = tf.nn.softmax(x)
        return x

    @tf.function
    def predict(self, inputs):      #输出行动
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)

if __name__ == '__main__':
    env = env_qoe(config)       # 实例化一个环境
    #########准确率测试
    accuracy = []
    constrain = []
    # accuracy_tar = []
    ###############实例化模型和优化器####
    ######创建新模型训练#####
    model = QNetwork()
    target_net = Target_QNetwork()
    weights = model.get_weights()
    target_net.set_weights(weights)
    ######读取模型继续训练#####
    # model = QNetwork()
    # target_net = Target_QNetwork()

    # model.load_weights('./checkpoints/my_checkpoint')
    # target_net.load_weights('./checkpoints/target_weight')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    ###########读取训练集###########
    # tra = np.load('dataset.npy')
    tra = np.load('dataset_python.npy')
    train_set = tra.tolist()
    train_set_len = len(train_set)
    #######创建经验池#######
    pool = Exp_pool(config)
    ######################
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        begin_time_all = time()
        #######每幕开始时，更新target_net参数######
        if episode_id > 200 and episode_id % 3 == 0:
            weights = model.get_weights()
            target_net.set_weights(weights)
        #######################################
        done = 0  # 本幕即将结束标识符

        time_record = 0 ###时间记录###
        #######训练集循环训练#######
        state1 = train_set[episode_id % train_set_len]
        c = state1[0:c_num]
        ####RNN补齐####
        # for i in range(121-105):
        #     c.append(0)
        ################
        state = env.reset(c)             # 初始化环境，获得初始状态
        ####计算当前探索率####
        #曲线一
        epsilon = max(
            initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
            final_epsilon)
        # 曲线二
        # tuo_a = num_exploration_episodes
        # tuo_b = 1 - final_epsilon
        # epsilon = max(
        #     pow((1 - pow(episode_id, 2) / pow(tuo_a, 2)) * pow(tuo_b, 2), 0.5).real + final_epsilon,
        #     final_epsilon)

        stop = 0
        step_count = 0
        # for t in range(group_of_x):   #每幕步数等于x组数
        #####程序各部分时间占比统计#####
        all_time = 0
        batch_time = 0
        decent_time = 0
        firstpart_time = 0

        while(stop != 1 and step_count <= 70):  ###幕一直进行到x选完或者达到上限
            step_count += 1

            firstpart_time_begin = time()
            if random.random() < epsilon:               # epsilon-greedy 探索策略，以 epsilon 的概率选择随机动作
                action = env.random_action(state)     # 选择随机动作（探索）on)
            else:
                action = model.predict(np.expand_dims(state[0], axis=0)).numpy()   # 选择模型计算出的 Q Value 最大的动作
                action = action[0]
            # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
            s_temp = copy.deepcopy(state)           #未知问题，state经过下面三行代码后值发生改变
            move = env.move(s_temp,action)
            reward = move[1]
            aaaaa = env.wrong_choose(state[0], action)
            if (aaaaa == 1):   #针对神经网络输出不符合约束，给予惩罚
                reward = -10000
            next_state = move[0]

            # 将(state, action, reward, next_state)的四元组（外加 done 标签表示是否结束）放入经验回放池

            #####判定幕是否即将结束###
            # label = next_state[1]
            # b = np.where(np.array(label) == 1)
            # if (len(b) == group_of_x-1):                                    # 幕即将结束标记
            #     done = 1
            # if (t == group_of_x-2):                                    # 幕即将结束标记
            #     done = 1
            if (next_state[1] == [1]*group_of_x):
                stop = 1
                done = 1
            ###########向经验池添加项目()##########
            ######
            # if(aaaaa != 1):
            pool.add(state[0], action, reward, next_state[0],done)
            ####################
            # 直到满足约束才会更新当前 state
            if (aaaaa == 0):
                state = copy.deepcopy(next_state)
            # if (t == group_of_x-1):                                    # 幕结束标记
            #     print("幕数 %4d  已完成,epsilon值为 %.4f," % (episode_id, epsilon))
            #if len(replay_buffer) >= batch_size:
            firstpart_time_end = time()

            if pool.count >= batch_size:

                begin_time_batch = time()
                # 从经验回放池中随机取一个批次的四元组
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = pool.get_batch()
                end_time_batch = time()

                begin_time_decent = time()


                ##########依次判断target_net对每个batch的输出是否满足约束，不满足的项目剔除，不参与损失函数计算
                # target_net_action = np.argmax(q_value, axis=1)
                # ti_01 = np.empty(batch_size)
                # for ti in range(batch_size):
                #     ti_01[ti] = env.wrong_choose(batch_next_state[ti], target_net_action[ti])
                # ti_index = np.where(np.array(ti_01) == 1)
                # ##剔除操作##
                # batch_state = np.delete(batch_state, ti_index, axis=0)
                # batch_action = np.delete(batch_action, ti_index, axis=0)
                # batch_reward = np.delete(batch_reward, ti_index, axis=0)
                # batch_next_state = np.delete(batch_next_state, ti_index, axis=0)
                # batch_done = np.delete(batch_done, ti_index, axis=0)
                # # q_value = np.delete(q_value, ti_index, axis=0)
                # q_value = model(batch_next_state)

                #####普通DQN#####
                target_q_value = target_net(batch_next_state)

                y = batch_reward + (gamma * tf.reduce_max(target_q_value, axis=1)) * (1 - batch_done)  # 计算 y 值
                #####Double DQN#####
                # q_max_a = tf.argmax(model(batch_next_state),axis=1)
                # q_target = target_net.CAQ(batch_next_state,q_max_a)
                #
                # y = batch_reward + (gamma * q_target) * (1 - batch_done)  # 计算 y 值


                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距离
                        y_true=y,
                        y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=x_num), axis=1)
                    )
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # 计算梯度并更新参数
                end_time_decent = time()

                batch_time += end_time_batch - begin_time_batch
                decent_time += end_time_decent - begin_time_decent
                firstpart_time += firstpart_time_end - firstpart_time_begin
        # print('测试代码块运行时间：', time_record)
    ##############每幕结束后，过一遍网络，测试准确率
        state = env.reset(c)
        fov_count = env.fov_count(state)
        for t in range(group_of_x):
            action = model.predict(np.expand_dims(state[0], axis=0)).numpy()  # 选择模型计算出的 Q Value 最大的动作
            action = action[0]
            s_temp = copy.deepcopy(state)
            move = env.move(s_temp, action)
            next_state = move[0]
            state = copy.deepcopy(next_state)

        # state = c + [0,0,0,1]*group_of_x
        # state = [state,[0]*group_of_x]               #已选择的x区域标记

        test_J = env.call(state)/fov_count  #######计算按可见的切块数量平均后的QOE值#######
        # record_J = test_J/max_J
        record_J = test_J
        accuracy.append(record_J)

        # print('幕总运行时间：', end_time_all-begin_time_all)
        #######约束满足图像#######
        b = 0
        b = np.where(np.array(state[1]) == 1)

        # tf.keras.backend.clear_session()
        end_time_all = time()
        # for t in range(group_of_x):
        #     action1 = target_net.predict(np.expand_dims(state[0], axis=0)).numpy()  # 选择模型计算出的 Q Value 最大的动作
        #     action1 = action1[0]
        #     s_temp = copy.deepcopy(state)
        #     move = env.move(s_temp, action1)
        #     next_state = move[0]
        #     state = copy.deepcopy(next_state)
        # test_J1 = env.call(state)
        # # record_J = test_J/max_J
        # record_J1 = test_J1
        # accuracy_tar.append(record_J1)
        time_record = end_time_all - begin_time_all
        # print('训练进度：',np.round(episode_id/num_episodes*100,4),'%','  ','第',episode_id,'幕J：',np.round(record_J,1))
        print('DDQN进度：',np.round(episode_id/num_episodes*100,4),'%','  ','第',episode_id,'幕：',b[0],np.round(record_J,1))
        print('时间统计：',' step数：',step_count,' 幕总运行时间:',time_record,' 读batch占比：',np.round(batch_time/time_record*100,1),'%',
              ' 梯度下降占比:',np.round(decent_time/time_record*100,1),'%',' first占比:',np.round(firstpart_time/time_record*100,1),'%'
              ,' 冗余占比:',np.round((time_record-batch_time-decent_time-firstpart_time)*100,1),'%')
#########输出结果###########
    # state = env.reset(c)
    # for t in range(group_of_x):
    #     action = model.predict(np.expand_dims(state[0], axis=0)).numpy()  # 选择模型计算出的 Q Value 最大的动作
    #     action = action[0]
    #     s_temp = copy.deepcopy(state)
    #     move = env.move(s_temp, action)
    #
    #     next_state = move[0]
    #     print("第",t,"步:")
    #     print(state)
    #     print(action)
    #     print(next_state)
    #     state = copy.deepcopy(next_state)
###########训练过程绘图#########


    mm = np.array(accuracy)
    np.save('J_track_DDQN_clean.npy', mm)
    tf.saved_model.save(model, 'DDQN_model_clean')

    model.save_weights('./DDQN_model/DDQN_model_variables/model')
    target_net.save_weights('./DDQN_model/DDQN_model_variables/target_weight')



    plt.figure(1)
    # plt.title('测试某些函数')
    # plt.ylim(ymin=-2000, ymax=0)
    plt.xlabel('number of episode')
    plt.plot(accuracy)
    # plt.plot(constrain)
    plt.ylabel('value of J')

    # plt.figure(2)
    # # plt.title('测试某些函数')
    # # plt.ylim(ymin=-2000, ymax=0)
    # plt.xlabel('number of episode')
    # plt.plot(accuracy_tar)
    # plt.ylabel('value of J_tar')

    plt.show()