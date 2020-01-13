# 地图划分
from pysc2.agents.myAgent.myAgent_12.config import config


ROW = 3
COLUMN = 3
ROW_DISTANCE = config.MAP_SIZE / ROW
COLUMN_DISTANCE = config.MAP_SIZE / COLUMN
MAP_ZONE = ROW * COLUMN


ACTION_DIM = MAP_ZONE * MAP_ZONE
STATE_DIM = 52
