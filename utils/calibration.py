import cv2 as cv 
import numpy as np

def get_CB_corners(frame, pattern_size, parameters=[(9, 9), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 100, 1e-6)]):
    window_size = parameters[0]
    term_criteria = parameters[1]
    g_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    pattern_frame = frame.copy()
    ret, corners = cv.findChessboardCorners(g_frame, pattern_size)
    if ret:
        cv.cornerSubPix(g_frame, corners, window_size, (-1, -1), term_criteria)
        # display chessboard pattern on images
        pattern_frame = cv.drawChessboardCorners(frame, pattern_size, corners, ret)
    return ret, pattern_frame, corners