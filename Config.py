class Config:
    ######QOE参数######
    N = 5
    tile_num = 8  # 切块数量
    level_num = 4  # 质量等级数量
    group_of_x = N * tile_num  # 决策变量x的组数（5个一组）

    x_num = level_num * group_of_x
    QSP_num = x_num
    fov_num = group_of_x
    fov_1_num = tile_num
    x_1_num = tile_num * level_num
    bf0_num = 1
    Dkc_num = group_of_x
    BW_num = N

    #c_num = 3*QSP_num + fov_num + fov_1_num + x_1_num + bf0_num + Dkc_num
    c_num = 2 * QSP_num + fov_num + fov_1_num + x_1_num + bf0_num + Dkc_num + BW_num
    s_len = x_num + c_num
    #######DQN参数#######
    num_episodes = 50  # 训练的总episode数量
    num_exploration_episodes = 145000  # 探索过程所占的episode数量
    # max_len_episode = 1500          # 每个episode的最大回合数
    memory_size = 130000    #池大小
    batch_size = 256  # 批次大小
    learning_rate = (1e-3)*1.5# 学习率
    gamma = 1.  # 折扣因子
    initial_epsilon = 1.  # 探索起始时的探索率
    final_epsilon = 0.01  # 探索终止时的探索率
