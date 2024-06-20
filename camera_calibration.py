import numpy as np
import cv2 as cv
import os

def single_camera_calibrate(cam_id, count, pattern_size=(7, 7)):
    # read all the images
    dir = 'single_images/{}'.format(cam_id)
    if not os.path.exists(dir):
        print("Directory does not exists, capture images first")
        return
    

    wp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    wp[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)


    images = ['{}/camera{}_{}.png'.format(dir, cam_id, cap_count) for cap_count in range(count)]
    world_points = []
    height, width = 0, 0
    display_frames = []
    for idx, image in enumerate(images):
        frame = cv.imread(image)
        height = frame.shape[0]
        width = frame.shape[1]
        g_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # find chessboard corners
        ret, corners = cv.findChessboardCorners(g_frame, pattern_size)
        if ret:
            cv.cornerSubPix(g_frame, corners, (9, 9), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 100, 1e-6))
            # display chessboard pattern on images
            pattern_frame = cv.drawChessboardCorners(frame, pattern_size, corners, ret)
            start_frame = cv.putText(pattern_frame, "Press space to find chessboard corners; s to skip to skip current frame; q to quit; Index: {}".format(idx),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            display_frames.append((start_frame, corners))


    index = 0
    start = False
    selected_corners = []
    while True:
        if index >= count or index >= len(display_frames):
            break
        cv.imshow("Frames", display_frames[index][0])
        key = cv.waitKey(1)
        if key & 0xFF == ord(' '):
            selected_corners.append(display_frames[index][1])
            world_points.append(wp)
            index += 1
        
        if key & 0xFF == ord('s'):
            print('skipped: {}'.format(index))
            index += 1
        
        if key & 0xFF == ord('q'):
            break
        
    cv.destroyAllWindows()
    # calculate intrinsic params
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_points, selected_corners, (width, height), None, None)
    print("{} - RMSE: ".format(cam_id), ret)
    print(mtx)
    # save intrinsic params
    np.savez('intrinsics/camera_calibration-{}'.format(cam_id), calibration_mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    return
