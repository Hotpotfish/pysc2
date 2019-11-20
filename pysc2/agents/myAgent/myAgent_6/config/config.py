
MAP_SIZE = 128

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.7  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 2000 # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch

MU = 0
SIGMA = 1
LEARING_RATE = 1e-2

MODEL_SAVE_EPISODE = 20
LOOP = 4