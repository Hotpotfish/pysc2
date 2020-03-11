MAP_SIZE = 64

FEATURE_UNITS_LENGTH = 29
RAW_UNITS_LENGTH = 46
EPISODES = 50000

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 600000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch

# 神经网络初始化参数
MU = 0
SIGMA = 1
TAU = 0.1
K = 1

LEARING_RATE = 1e-1
# 模型保存轮次
MODEL_SAVE_EPISODE = 1000

# sub plt
ROW = 4
COLUMN = 4

MY_UNIT_NUMBER = 8
ENEMY_UNIT_NUMBER = 4  # 选择敌方智能体的空间

ACTION_DIM = 1  # （4为上下左右）

COOP_AGENT_OBDIM = 8
COOP_AGENTS_OBDIM = (MY_UNIT_NUMBER + ENEMY_UNIT_NUMBER) * COOP_AGENT_OBDIM

OB_RANGE = 80
