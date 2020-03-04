import matplotlib.pyplot as plt
import numpy as np


def read_and_plot_reward(path):
    f = open(path, 'r')
    points = []
    point = f.readline().strip('\n').split(' ')
    while 1:

        if len(point) != 2:
            break
        # print(point)
        x = float(point[0])
        y = float(point[1])
        points.append([x, y])
        point = f.readline().strip('\n').split(' ')

    f.close()
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1])

    plt.xlabel('epsoide')
    plt.ylabel('reward')
    plt.title('reward curve')
    plt.show()

def read_and_plot_win_rate(path):
    f = open(path, 'r')
    points = []
    point = f.readline().strip('\n').split(' ')
    while 1:

        if len(point) != 2:
            break
        # print(point)
        x = float(point[0])
        y = float(point[1])
        points.append([x, y])
        point = f.readline().strip('\n').split(' ')

    f.close()
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1])

    plt.xlabel('epsoide')
    plt.ylabel('win_rate')
    plt.title('win_rate curve')
    plt.show()


def plot_all(path):
    read_and_plot_reward(path + '/BIC_DDPG_reward.txt')
    read_and_plot_win_rate(path + '/BIC_DDPG_win_rate.txt')


if __name__ == "__main__":
    plot_all('d:/model/Bic-DDPG_mix_2')

# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.signal
#
#
# def read_and_plot(path):
#     f = open(path, 'r')
#     points = []
#     point = f.readline().strip('\n').split(' ')
#     while 1:
#
#         if len(point) != 2:
#             break
#         # print(point)
#         x = float(point[0])
#         y = float(point[1])
#         points.append([x, y])
#         point = f.readline().strip('\n').split(' ')
#
#     f.close()
#     points = np.array(points)
#     plt.plot(points[:, 0], scipy.signal.savgol_filter(points[:, 1], 51, 2, mode='nearest'))
#
#     plt.xlabel('epsoide')
#     plt.ylabel('reward')
#     plt.title('reward curve')
#     plt.show()
#
#
# if __name__ == "__main__":
#     read_and_plot('d:/model/20200203155334/reward.txt')
