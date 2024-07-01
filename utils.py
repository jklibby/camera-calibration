import cv2 as cv
import numpy as np

def checkerboard_orientation(corners):
    corners = corners.reshape((-1, 2))
    corner_1 = corners[0]
    corner_2 = corners[1]
    x_norm_ball = np.linalg.norm(corner_1[0] - corner_2[0])
    y_norm_ball = np.linalg.norm(corner_1[1] - corner_2[1])

    if x_norm_ball > y_norm_ball:
        if corner_1[0] > corner_2[0]:
            return "x_1"
        else:
            return "x_2"
    else:
        if corner_1[1] > corner_2[1]:
            return "y_1"
        else:
            return "y_2"