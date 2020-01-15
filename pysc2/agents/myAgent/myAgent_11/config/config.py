MAP_SIZE = 64

FEATURE_UNITS_LENGTH = 29
RAW_UNITS_LENGTH = 46

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
REPLAY_SIZE = 1000000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch

# 神经网络初始化参数
MU = 0
SIGMA = 1
LEARING_RATE = 5e-4
# 模型保存轮次
MODEL_SAVE_EPISODE = 1000

# sub plt
ROW = 4
COLUMN = 4

# 命令参数的最大长度
ORDERLENTH = 5
# K = 3

# 参数维度 = 队列（QUEUED）+我方目标单位(MY_UNIT_NUMBER)+敌方目标单位(ENEMY_UNIT_NUMBER) +坐标(MAP_SIZE *MAP_SIZE)
QUEUED = 2
MY_UNIT_NUMBER = 3
ENEMY_UNIT_NUMBER = 3# 选择敌方智能体的空间
POINT_NUMBER = MAP_SIZE * MAP_SIZE

STATIC_ACTION_DIM = 1 + 4

ATTACT_CONTROLLER_ACTIONDIM = STATIC_ACTION_DIM + ENEMY_UNIT_NUMBER  # （4为上下左右）
ATTACT_CONTROLLER_ACTIONDIM_BIN = len('{:b}'.format(ATTACT_CONTROLLER_ACTIONDIM))

COOP_AGENTS_OBDIM = (MY_UNIT_NUMBER + ENEMY_UNIT_NUMBER) * 7

GAMMA_FOR_UPDATE = 0.1
OB_RANGE = 6
ATTACK_RANGE = 4
