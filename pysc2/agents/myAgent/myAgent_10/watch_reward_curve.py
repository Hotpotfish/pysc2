import matplotlib.pyplot as plt
import numpy as np


def read_and_plot(path):
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


if __name__ == "__main__":
    read_and_plot('d:/model/20191231011353/reward.txt')
