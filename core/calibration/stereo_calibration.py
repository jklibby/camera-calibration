import numpy as np
import cv2 as cv
import os
from pathlib import Path

from options import StereoCameraCalibrationOptions, StereoCameraRectificationOptions
from utils.calibration import get_CB_corners
from visualization import visualize_stereo_errors, Thresholder


def stereo_camera_calibrate(opts: StereoCameraCalibrationOptions) -> float:
    """
        Function to calibrate a stereo camera. Detects checkerboard patterns in captured
        paired images. Uses the detected pattern to calculate extrinsic params, 
        Rotation matrix, Translation vector, Essential matrix, Fundamental matrix and repreojection error. 
        Plots the reprojection error if `opts.headless` is `False`. Eliminates patterns which lie above
        `opts.error_threshold` or the selected threshold in the matplotlib reprojection error plot. 

        Args:
            opts (StereoCameraCalibrationOptions): Options for calibrating a stereo camera
        
        Outputs:
            stores extrinsic params in the `opts.dir/extrinsic_params/stereo_calibration.npz`
        
        Returns:
            Reprojection error (float): Final stereo reprojection error after discarding outliers.
    """
    extrinsics_dir = Path(opts.dir)
    intrinsics_dir = Path(opts.intrinsics_dir)
    if not Path.exists(extrinsics_dir):
        Path.mkdir(extrinsics_dir, parents=True)
    
    frame_pair_dir = opts.paired_images_path
    if not os.path.exists(frame_pair_dir):
        raise Exception("Paired images directory not found")
    criteria = opts.criteria

    pattern_size = opts.pattern_size
    wp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    wp[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)
    wp *= 1.5

    left_cam, right_cam = opts.left_cam_id, opts.right_cam_id
    count = opts.count
    images = opts.load_paired_images()
    world_points = []
    image_size = []
    left_display_frames = []
    right_display_frames = []
    for index, image in enumerate(images):
        left_frame = cv.imread(image[0])
        right_frame = cv.imread(image[1])
        height, width = (left_frame.shape[0], left_frame.shape[1])
        image_size = [width, height]
        
        left_ret, left_pattern_frame, left_corners = get_CB_corners(left_frame, pattern_size)
        right_ret, right_pattern_frame,  right_corners = get_CB_corners(right_frame, pattern_size)
        if left_ret and right_ret:
            left_start_frame = cv.putText(left_pattern_frame, "Press space to accept chessboard corners; Press p to accept all; s to skip to skip current frame; q to quit; Index: {}".format(index),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            right_start_frame = cv.putText(right_pattern_frame, "Press space to accept chessboard corners; Press p to accept all; s to skip to skip current frame; q to quit; Index: {}".format(index),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            
            left_display_frames.append((left_start_frame, left_corners))
            right_display_frames.append((right_start_frame, right_corners))

    if not opts.headless:
        opts.cv_options.named_window("Left Frame")
        opts.cv_options.named_window("Right Frame")

    left_selected_corners = []
    right_selected_corners = []
    selected_index = []
    index = 0
    skip_all = True
    while True:
        if (not opts.headless) and skip_all:
            cv.imshow("Left Frame", left_display_frames[index][0])
            cv.imshow("Right Frame", right_display_frames[index][0])
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
            
            if key & 0xFF == ord('p'):
                skip_all = False
                cv.destroyAllWindows()
            
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

    left_intrinsics_path = str(intrinsics_dir.joinpath("camera_calibration_{}.npz".format(opts.left_cam_id)).absolute())
    right_intrinsics_path = str(intrinsics_dir.joinpath("camera_calibration_{}.npz".format(opts.right_cam_id)).absolute())
    intrinsic_data_0 = np.load(left_intrinsics_path)
    intrinsic_data_1 = np.load(right_intrinsics_path)
    left_cam_mtx = intrinsic_data_0['calibration_mtx']
    left_dist = intrinsic_data_0['dist']
    right_cam_mtx = intrinsic_data_1['calibration_mtx']
    right_dist = intrinsic_data_1['dist']

    ret, cm1, dist1, cm2, dist2, R, T, E, F = cv.stereoCalibrate(world_points, left_selected_corners,  right_selected_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, criteria=criteria, flags=opts.flags)
    ret, cm1, dist1, cm2, dist2, R, T, E, F, rvecs, tvecs, per_view_errors = cv.stereoCalibrateExtended(world_points, left_selected_corners,  right_selected_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, R, T, criteria=criteria, flags=opts.flags)
        
    print("Stereo RMSE: ", ret)
    print("Baseline: {0:3f} cm | ".format(np.linalg.norm(T)))
    stereo_calibration_path = str(extrinsics_dir.joinpath("stereo_calibration").absolute())
    np.savez(stereo_calibration_path, RMSE=ret, R=R, T=T, E=E, F=F, image_size=image_size)

    thresholder = Thresholder(opts.error_threshold)
    if not opts.headless:
        visualize_stereo_errors(thresholder, per_view_errors)
    
    indexes = np.arange(len(per_view_errors)).reshape(-1, 1)
    per_view_errors = np.hstack([per_view_errors, indexes])
    # refine stereo calibration by excluding calibration with high error
    pi_1 = per_view_errors[per_view_errors[:, 0] < thresholder.threshold]
    pi = pi_1[pi_1[:, 1] < thresholder.threshold]
    indexes_below_threshold = np.array(pi[:, 2], dtype=np.uint8)
    
    if not opts.headless:
        index = 0 
        opts.cv_options.named_window("Left Frame")
        opts.cv_options.named_window("Right Frame")

        while index < len(indexes_below_threshold):
            left_error_frame = left_display_frames[indexes_below_threshold[index]][0]
            right_error_frame = right_display_frames[indexes_below_threshold[index]][0]
            left_error_frame = cv.putText(left_error_frame, "Error: {}".format(per_view_errors[indexes_below_threshold[index]][:2]), (100, 200), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            right_error_frame = cv.putText(right_error_frame, "Error: {}".format(per_view_errors[indexes_below_threshold[index]][:2]), (100, 200), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            cv.imshow("Right Frame", right_error_frame)
            cv.imshow("Left Frame", left_error_frame)
            key  = cv.waitKey(1)
            if key & 0xFF == ord(' '):
                index += 1
            if key & 0xFF == ord('q'):
                break
        
        cv.destroyAllWindows()

    print("Indexes below threshold: ", len(indexes_below_threshold))
    if len(indexes_below_threshold) < 15:
        print("Not enough images to calculate refined stereo")
        return ret
    left_selected_corners = np.array(left_selected_corners)
    right_selected_corners = np.array(right_selected_corners)
    world_points = np.array(world_points)

    left_refined_corners, right_refined_corners, world_refined_corners = left_selected_corners[indexes_below_threshold, :], right_selected_corners[indexes_below_threshold, :], world_points[indexes_below_threshold, :]
    refined_ret, cm1, dist1, cm2, dist2, R, T, E, F = cv.stereoCalibrate(world_refined_corners, left_refined_corners,  right_refined_corners, left_cam_mtx, left_dist, right_cam_mtx, right_dist, image_size, criteria=criteria, flags=opts.flags)
    print("Refined Stereo RMSE: ", refined_ret)
    print("Baseline: {0:3f} cm | ".format(np.linalg.norm(T)))
    if refined_ret < ret:
        ret = refined_ret
        np.savez(stereo_calibration_path, RMSE=ret, R=R, T=T, E=E, F=F, image_size=image_size)

    return ret


def stereo_rectification(opts: StereoCameraRectificationOptions):
    """
        Function to calculate rectification maps for stereo camera based on the estrinsic 
        parmaters. If `opts.headless` is `False` then display rectified images and save images.

        Args:
            opts (StereoCameraRectificationOptions): Options for rectification a stereo camera
        
        Outputs:
            stores recitification maps in the `opts.dir/extrinsic_params/stereo_rectification/stereo_rectification_maps.npz`
        
        Returns:
            Reprojection error (float): Final stereo reprojection error after discarding outliers.
    """
    intrinsics_dir = Path(opts.intrinsic_dir)
    extrinsics_dir = Path(opts.extrinsic_dir)
    rectified_dir = extrinsics_dir.joinpath("stereo_rectification")
    if not Path.exists(rectified_dir):
        Path.mkdir(rectified_dir, parents=True)
    
    rectified_images_dir = rectified_dir.joinpath("rectified_images")
    if not Path.exists(rectified_images_dir):
        Path.mkdir(rectified_images_dir, parents=True)
    

    left_intrinsics_path = str(intrinsics_dir.joinpath("camera_calibration_{}.npz".format(opts.left_cam_id)).absolute())
    right_intrinsics_path = str(intrinsics_dir.joinpath("camera_calibration_{}.npz".format(opts.right_cam_id)).absolute())
    
    intrinsic_data_0 = np.load(left_intrinsics_path)
    intrinsic_data_1 = np.load(right_intrinsics_path)
    left_cam_mtx = intrinsic_data_0['calibration_mtx']
    left_dist = intrinsic_data_0['dist']
    right_cam_mtx = intrinsic_data_1['calibration_mtx']
    right_dist = intrinsic_data_1['dist']

    # load rotation and trnaslation
    stereo_calibration_path = str(extrinsics_dir.joinpath("stereo_calibration.npz").absolute())
    stereo_calibration = np.load(stereo_calibration_path)
    # perform stereo calibration
    R, T, E, F, image_size = stereo_calibration['R'], stereo_calibration['T'], stereo_calibration['E'], stereo_calibration['F'], stereo_calibration['image_size']
    left_rect, right_rect,  left_proj, right_proj, Q, left_roi, right_roi = cv.stereoRectify(
        left_cam_mtx, 
        left_dist, 
        right_cam_mtx, 
        right_dist, 
        image_size, 
        R, T,
        None, None, None, None, 
        alpha=0,
        flags=opts.flags
    )
    print("rectification image size", image_size)
    left_map_x, left_map_y = cv.initUndistortRectifyMap(left_cam_mtx, left_dist, left_rect, left_proj, image_size, cv.CV_32FC1)
    right_map_x, right_map_y = cv.initUndistortRectifyMap(right_cam_mtx, right_dist, right_rect, right_proj, image_size, cv.CV_32FC1)

    stereo_remaps_path = rectified_dir.joinpath("stereo_rectification_maps")
    np.savez(stereo_remaps_path, left_map_x=left_map_x, left_map_y=left_map_y, right_map_x=right_map_x, right_map_y=right_map_y)

    if opts.headless:
        return
    images = opts.load_paired_images()
    index = 0
    left_frame = cv.imread(images[index][0])
    right_frame = cv.imread(images[index][1])
    # TO-DO: Load interpolation from a config file
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
    
    opts.cv_options.named_window("Rectified Image")
    while True:
        cv.imshow("Rectified Image", rect_image)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            left_recitfied_image_path = str(rectified_images_dir.joinpath("camera{}_{}.png".format(opts.left_cam_id, index)))
            right_recitfied_image_path = str(rectified_images_dir.joinpath("camera{}_{}.png".format(opts.right_cam_id, index)))
            cv.imwrite(left_recitfied_image_path, left_frame_rect)
            cv.imwrite(right_recitfied_image_path, right_frame_rect)
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


