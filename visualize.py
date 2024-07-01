import matplotlib.pyplot as plt
import numpy as np

def visualize_pcd(points):
    points = points.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    plt.show()


