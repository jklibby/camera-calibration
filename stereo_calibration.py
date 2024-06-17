import numpy as np
import cv2 as cv
from pprint import pprint


def stereo_camera_calibrate(count, pattern_size=(7, 7)):
    frame_pair_dir = 'paired_images'
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    wp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    wp[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)


    images = [('{}/camera0_{}.png'.format(frame_pair_dir, cap_count), '{}/camera1_{}.png'.format(frame_pair_dir, cap_count)) for cap_count in range(count)]
    world_points = []
    image_size = []
    left_display_frames = []
    right_display_frames = []
    for index, image in enumerate(images):
        left_frame = cv.imread(image[0])
        right_frame = cv.imread(image[1])
        image_size = (left_frame.shape[1], left_frame.shape[0])
        left_g_frame = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)
        right_g_frame = cv.cvtColor(right_frame, cv.COLOR_BGR2GRAY)

        left_ret, left_corners = cv.findChessboardCorners(left_g_frame, pattern_size)
        right_ret, right_corners = cv.findChessboardCorners(right_g_frame, pattern_size)
        if left_ret and right_ret:
            cv.cornerSubPix(left_g_frame, left_corners, (11, 11), (-1, -1), criteria)
            cv.cornerSubPix(right_g_frame, right_corners, (11, 11), (-1, -1),criteria)
            left_pattern_frame = cv.drawChessboardCorners(left_frame, pattern_size, left_corners, left_ret)
            right_pattern_frame = cv.drawChessboardCorners(right_frame, pattern_size, right_corners, right_ret)
            left_start_frame = cv.putText(left_pattern_frame, "Press space to find chessboard corners; s to skip to skip current frame; q to quit; Index: {}".format(index),(100, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, 1)
            right_start_frame = cv.putText(right_pattern_frame, "Press space to find chessboard corners; s to skip to skip current frame; q to quit; Index: {}".format(index),(100, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, 1)
            left_display_frames.append((left_start_frame, left_corners))
            right_display_frames.append((right_start_frame, right_corners))

    left_selected_corners = []
    right_selected_corners = []
    index = 0
    while True:
        cv.imshow("Left Frames", left_display_frames[index][0])
        cv.imshow("Right Frames", right_display_frames[index][0])
        key = cv.waitKey(1)
        if key & 0xFF == ord(' '):
            left_selected_corners.append(left_display_frames[index][1])
            right_selected_corners.append(right_display_frames[index][1])
            world_points.append(wp)
            index += 1
        
        if key & 0xFF == ord('s'):
            print('skipped: {}'.format(index))
            index += 1
        
        if key & 0xFF == ord('q'):
            break

        if index >= count or index >= len(left_display_frames):
            break
    
    cv.destroyAllWindows()

    intrinsic_data_0 = np.load('intrinsics/camera_calibration-0.npz')
    intrinsic_data_1 = np.load('intrinsics/camera_calibration-1.npz')
    left_cam_mtx = intrinsic_data_0['calibration_mtx']
    left_dist = intrinsic_data_0['dist']
    right_cam_mtx = intrinsic_data_1['calibration_mtx']
    right_dist = intrinsic_data_1['dist']

    # new_left_mtx, _ = cv.getOptimalNewCameraMatrix(left_cam_mtx, left_dist, image_size, 1, image_size)
    # new_right_mtx, _ = cv.getOptimalNewCameraMatrix(right_cam_mtx, right_dist, image_size, 1, image_size)
    
    flags = cv.CALIB_FIX_INTRINSIC
    ret, cm1, dist1, cm2, dist2, R, T, E, F = cv.stereoCalibrate(world_points, left_selected_corners, right_selected_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, criteria=criteria, flags=flags)
    print(ret)
    np.savez('stereo_calibration', R=R, T=T, E=E, F=F, image_size=image_size)
    return


def stereo_rectification(count, pattern_size=(7, 7)):
    intrinsic_data_0 = np.load('intrinsics/camera_calibration-0.npz')
    intrinsic_data_1 = np.load('intrinsics/camera_calibration-1.npz')
    left_cam_mtx = intrinsic_data_0['calibration_mtx']
    left_dist = intrinsic_data_0['dist']
    right_cam_mtx = intrinsic_data_1['calibration_mtx']
    right_dist = intrinsic_data_1['dist']

    # load rotation and trnaslation
    extrinsics = np.load('stereo_calibration.npz')
    # perform stereo calibration
    R, T, E, F, image_size = extrinsics['R'], extrinsics['T'], extrinsics['E'], extrinsics['F'], extrinsics['image_size']
    left_rect, right_rect,  left_proj, right_proj, Q, left_roi, right_roi = cv.stereoRectify(
        left_cam_mtx, 
        left_dist, 
        right_cam_mtx, 
        right_dist, 
        image_size, 
        R, T,
        None, None, None, None, 
        flags=cv.CALIB_ZERO_DISPARITY, alpha=1
    )
    left_map_x, left_map_y = cv.initUndistortRectifyMap(left_cam_mtx, left_dist, left_rect, left_proj, image_size, cv.CV_32FC1)
    right_map_x, right_map_y = cv.initUndistortRectifyMap(right_cam_mtx, right_dist, right_rect, right_proj, image_size, cv.CV_32FC1)

    np.savez("stereo-rectified-maps", left_map_x=left_map_x, left_map_y=left_map_y, right_map_x=right_map_x, right_map_y=right_map_y)

    frame_pair_dir = 'paired_images'
    images = [('{}/camera0_{}.png'.format(frame_pair_dir, cap_count), '{}/camera1_{}.png'.format(frame_pair_dir, cap_count)) for cap_count in range(count)]
    index = 0
    left_frame = cv.imread(images[index][0])
    right_frame = cv.imread(images[index][1])
    left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
    right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    while True:
        cv.imshow("Left Rectified Frame", left_frame_rect)
        cv.imshow("Right Rectified Frame", right_frame_rect)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord(' '):
            index += 1
            if index >= len(images):
                break
            left_frame = cv.imread(images[index][0])
            right_frame = cv.imread(images[index][1])
            left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
            right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    cv.destroyAllWindows()