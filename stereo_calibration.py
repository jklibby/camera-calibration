import numpy as np
import cv2 as cv
import os
from pprint import pprint

from utils import checkerboard_orientation


def stereo_camera_calibrate(count, path, pattern_size=(7, 7), flags=None, dir=None, skip_gui=False, refine_val=1):
    extrinsics_dir = "extrinsics"
    if dir:
        extrinsics_dir = "{}/{}".format(extrinsics_dir, dir)
    if not os.path.exists(extrinsics_dir):
        os.makedirs(extrinsics_dir)
    frame_pair_dir = path
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

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
            cv.cornerSubPix(left_g_frame, left_corners, (9, 9), (-1, -1), criteria)
            cv.cornerSubPix(right_g_frame, right_corners, (9, 9), (-1, -1), criteria)
            
            left_pattern_frame = cv.drawChessboardCorners(left_frame, pattern_size, left_corners, left_ret)
            right_pattern_frame = cv.drawChessboardCorners(right_frame, pattern_size, right_corners, right_ret)
            
            left_start_frame = cv.putText(left_pattern_frame, "Press space to find chessboard corners; s to skip to skip current frame; q to quit; Index: {}".format(index),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            right_start_frame = cv.putText(right_pattern_frame, "Press space to find chessboard corners; s to skip to skip current frame; q to quit; Index: {}".format(index),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            
            left_display_frames.append((left_start_frame, left_corners))
            right_display_frames.append((right_start_frame, right_corners))

    left_selected_corners = []
    right_selected_corners = []
    selected_index = []
    index = 0
    while True:
        if not skip_gui:
            cv.imshow("Left Frames", left_display_frames[index][0])
            cv.imshow("Right Frames", right_display_frames[index][0])
            key = cv.waitKey(1)
            if key & 0xFF == ord(' '):
                left_selected_corners.append(left_display_frames[index][1])
                right_selected_corners.append(right_display_frames[index][1])
                world_points.append(wp)
                selected_index.append(index)
                index += 1
            
            if key & 0xFF == ord('s'):
                print('skipped: {}'.format(index))
                index += 1
            
            if key & 0xFF == ord('q'):
                break
            
        else:
            left_selected_corners.append(left_display_frames[index][1])
            right_selected_corners.append(right_display_frames[index][1])
            world_points.append(wp)
            selected_index.append(index)
            index += 1
        if index >= count or index >= len(left_display_frames):
            break
    
    cv.destroyAllWindows()

    intrinsic_data_0 = np.load('intrinsics/{}/camera_calibration_0.npz'.format(dir))
    intrinsic_data_1 = np.load('intrinsics/{}/camera_calibration_1.npz'.format(dir))
    left_cam_mtx = intrinsic_data_0['calibration_mtx']
    left_dist = intrinsic_data_0['dist']
    right_cam_mtx = intrinsic_data_1['calibration_mtx']
    right_dist = intrinsic_data_1['dist']

    # new_left_mtx, _ = cv.getOptimalNewCameraMatrix(left_cam_mtx, left_dist, image_size, 1, image_size)
    # new_right_mtx, _ = cv.getOptimalNewCameraMatrix(right_cam_mtx, right_dist, image_size, 1, image_size)
    
    if flags:
        ret, cm1, dist1, cm2, dist2, R, T, E, F = cv.stereoCalibrate(world_points, left_selected_corners,  right_selected_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, criteria=criteria, flags=flags)
        ret, cm1, dist1, cm2, dist2, R, T, E, F, rvecs, tvecs, per_view_errors = cv.stereoCalibrateExtended(world_points, left_selected_corners,  right_selected_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, R, T, criteria=criteria, flags=flags)
        
    else:
        ret, cm1, dist1, cm2, dist2, R, T, E, F = cv.stereoCalibrate(world_points, left_selected_corners,  right_selected_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, criteria=criteria)
    print("Stereo RMSE: ", ret)
    print("Baseline: {0:3f} cm | ".format(np.linalg.norm(T) * 1.5))
    np.savez('{}/stereo_calibration'.format(extrinsics_dir), R=R, T=T, E=E, F=F, image_size=image_size)

    if refine_val < 0:
        return ret
    # per_view_errors = per_view_errors or []
    index = 0 
    # while index < len(per_view_errors):
    #     left_error_frame = left_display_frames[selected_index[index]][0]
    #     right_error_frame = right_display_frames[selected_index[index]][0]
    #     left_error_frame = cv.putText(left_error_frame, "Error: {}".format(per_view_errors[index]), (100, 200), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
    #     right_error_frame = cv.putText(right_error_frame, "Error: {}".format(per_view_errors[index]), (100, 200), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
    #     cv.imshow("Right Error Frame", right_error_frame)
    #     cv.imshow("Left Error Frame", left_error_frame)
    #     key  = cv.waitKey(1)
    #     if key & 0xFF == ord(' '):
    #         index += 1
    #     if key & 0xFF == ord('q'):
    #         break
    
    per_view_errors = np.array(per_view_errors)
    # refine stereo calibration by excluding calibration with high error
    pi_1 = per_view_errors[per_view_errors[:, 0] < 1]
    pi = pi_1[pi_1[:, 1] < 1]

    indexes = np.array([np.argwhere(per_view_errors[:, 0] == a[0]) for a in pi])
    indexes = indexes.reshape(-1)
    if len(indexes) < 15:
        print("Not enough images to calculate refined stereo")
        return ret
    left_selected_corners = np.array(left_selected_corners)
    right_selected_corners = np.array(right_selected_corners)
    world_points = np.array(world_points)

    left_refined_corners, right_refined_corners, world_refined_corners = left_selected_corners[indexes, :], right_selected_corners[indexes, :], world_points[indexes, :]
    print(left_refined_corners.shape, right_refined_corners.shape, world_refined_corners.shape)
    ret, cm1, dist1, cm2, dist2, R, T, E, F = cv.stereoCalibrate(world_refined_corners, left_refined_corners,  right_refined_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, criteria=criteria, flags=flags)
    print("Refined Stereo RMSE: ", ret)
    print("Baseline: {0:3f} cm | ".format(np.linalg.norm(T) * 1.5))
    np.savez('{}/stereo_calibration'.format(extrinsics_dir), R=R, T=T, E=E, F=F, image_size=image_size)

    return ret


def stereo_rectification(count, path, flags=(), dir=None, skip_gui=True):
    rectified_dir = 'rectified_images'
    if dir:
        rectified_dir = "{}/{}".format(rectified_dir, dir)
    if not os.path.exists(rectified_dir):
        os.makedirs(rectified_dir)
    intrinsic_data_0 = np.load('intrinsics/{}/camera_calibration_0.npz'.format(dir))
    intrinsic_data_1 = np.load('intrinsics/{}/camera_calibration_1.npz'.format(dir))
    left_cam_mtx = intrinsic_data_0['calibration_mtx']
    left_dist = intrinsic_data_0['dist']
    right_cam_mtx = intrinsic_data_1['calibration_mtx']
    right_dist = intrinsic_data_1['dist']

    # load rotation and trnaslation
    extrinsics = np.load('extrinsics/{}/stereo_calibration.npz'.format(dir))
    # perform stereo calibration
    if flags:
        R, T, E, F, image_size = extrinsics['R'], extrinsics['T'], extrinsics['E'], extrinsics['F'], extrinsics['image_size']
        left_rect, right_rect,  left_proj, right_proj, Q, left_roi, right_roi = cv.stereoRectify(
            left_cam_mtx, 
            left_dist, 
            right_cam_mtx, 
            right_dist, 
            image_size, 
            R, T,
            None, None, None, None, 
            flags=flags, alpha=1
        )
    else:
        R, T, E, F, image_size = extrinsics['R'], extrinsics['T'], extrinsics['E'], extrinsics['F'], extrinsics['image_size']
        left_rect, right_rect,  left_proj, right_proj, Q, left_roi, right_roi = cv.stereoRectify(
            left_cam_mtx, 
            left_dist, 
            right_cam_mtx, 
            right_dist, 
            image_size, 
            R, T,
            None, None, None, None, alpha=1
        )
    left_map_x, left_map_y = cv.initUndistortRectifyMap(left_cam_mtx, left_dist, left_rect, left_proj, image_size, cv.CV_32FC1)
    right_map_x, right_map_y = cv.initUndistortRectifyMap(right_cam_mtx, right_dist, right_rect, right_proj, image_size, cv.CV_32FC1)

    np.savez("extrinsics/{}/stereo-rectified-maps".format(dir), left_map_x=left_map_x, left_map_y=left_map_y, right_map_x=right_map_x, right_map_y=right_map_y)

    if skip_gui:
        return
    frame_pair_dir = path
    images = [('{}/camera0_{}.png'.format(frame_pair_dir, cap_count), '{}/camera1_{}.png'.format(frame_pair_dir, cap_count)) for cap_count in range(count)]
    index = 0
    left_frame = cv.imread(images[index][0])
    right_frame = cv.imread(images[index][1])
    left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
    right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    rect_image = np.hstack([left_frame_rect, right_frame_rect])
    height, width = left_frame.shape[:2]
    num_lines = 50
    interval = height // num_lines
    x = 0
    for i in range(num_lines):
        x = interval*i
        rect_image = cv.line(rect_image, (0, x), (width*2, x), (0, 255, 0))
    while True:
        cv.imshow("Rectified", rect_image)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            cv.imwrite('{}/camera0_{}.png'.format(rectified_dir, index), left_frame_rect)
            cv.imwrite('{}/camera1_{}.png'.format(rectified_dir, index), right_frame_rect)
        if key & 0xFF == ord(' '):
            index += 1
            if index >= len(images):
                break
            left_frame = cv.imread(images[index][0])
            right_frame = cv.imread(images[index][1])
            left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
            right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
            rect_image = np.hstack([left_frame_rect, right_frame_rect])
            x = 0
            for i in range(num_lines):
                x = interval*i
                rect_image = cv.line(rect_image, (0, x), (width*2, x), (0, 255, 0))
    cv.destroyAllWindows()



def stereo_reprojection_error():
    "Calculate the stereo reprojection for left camera and right camera"
    pass

