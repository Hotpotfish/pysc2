# 地图划分
import pysc2.agents.myAgent.myAgent_12.smart_actions as smart_actions
from pysc2.agents.myAgent.myAgent_12.config import config


ROW = 3
COLUMN = 3
ROW_DISTANCE = config.MAP_SIZE / ROW
COLUMN_DISTANCE = config.MAP_SIZE / COLUMN
MAP_ZONE = ROW * COLUMN


ACTION_DIM = 81
STATE_DIM = 52
