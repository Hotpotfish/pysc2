MAP_SIZE = 128

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.7  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 2048  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch

# 神经网络初始化参数
MU = 0
SIGMA = 1
LEARING_RATE = 1e-2
# 模型保存轮次
MODEL_SAVE_EPISODE = 20

# sub plt
ROW = 4
COLUMN = 4

#命令参数的最大长度
ORDERLENTH = 5


# 参数维度 = 队列（QUEUED）+我方目标单位(MY_UNIT_NUMBER)+敌方目标单位(ENEMY_UNIT_NUMBER) +坐标(MAP_SIZE *MAP_SIZE)
QUEUED = 2
MY_UNIT_NUMBER = 30
ENEMY_UNIT_NUMBER = 30
POINT_NUMBER = MAP_SIZE * MAP_SIZE


ATTACT_CONTROLLER_PARAMETERDIM = QUEUED + MY_UNIT_NUMBER + ENEMY_UNIT_NUMBER + MAP_SIZE * MAP_SIZE
