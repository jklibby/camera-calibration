import numpy as np
import cv2 as cv
from pathlib import Path

from options import SingleCameraCalibrateOptions
from utils.calibration import get_CB_corners
from projection import project_world_to_camera
from visualization import visualize_errors, Thresholder

def single_camera_calibrate(opts: SingleCameraCalibrateOptions) -> float:
    # read all the images
    intrinsics_dir = Path(opts.dir)
    if not Path(opts.path).absolute().exists():
       raise Exception("{} : Directory does not exists, capture images first".format(opts.path))
    
    if not Path.exists(intrinsics_dir):
        Path.mkdir(intrinsics_dir, parents=True)

    cam_id = opts.cam_id
    pattern_size = opts.pattern_size
    count = opts.count

    wp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    wp[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)
    wp *= 1.5

    images = opts.load_single_images()
    world_points = list()
    height, width = 0, 0
    display_frames = list()
    for idx, image in enumerate(images):
        frame = cv.imread(image)
        height = frame.shape[0]
        width = frame.shape[1]
        # find chessboard corners
        ret, pattern_frame, corners = get_CB_corners(frame, pattern_size)
        if ret:
            start_frame = cv.putText(pattern_frame, "Press space to find chessboard corners; s to skip to skip current frame; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            display_frames.append((start_frame, corners))

    if not opts.headless:
        frame_name = "Frames {}".format(cam_id)
        opts.cv_options.named_window(frame_name)
    selected_corners = list()
    curr_count = count
    index = 0
    skip_all = True
    while True:
        if curr_count <= 0 or index >= len(display_frames):
            break
        if (not opts.headless) and skip_all:
            cv.imshow(frame_name, display_frames[index][0])
            key = cv.waitKey(1)
            if key & 0xFF == ord(' '):
                selected_corners.append(display_frames[index][1])
                world_points.append(wp)
                index += 1
                curr_count -= 1

            if key & 0xFF == ord('s'):
                print('skipped: {}'.format(index))
                index += 1
            
            if key & 0xFF == ord('p'):
                skip_all = False
                cv.destroyAllWindows()
        
            if key & 0xFF == ord('q'):
                cv.destroyAllWindows()
                return -1
        else:
            selected_corners.append(display_frames[index][1])
            world_points.append(wp)
            index += 1
            curr_count -= 1
        
            
        
    cv.destroyAllWindows()
    # calculate intrinsic params
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_points, selected_corners, (width, height), None, None, flags=opts.flags, criteria=opts.criteria)
    
    print("{} - RMSE: ".format(cam_id), ret)
    re, per_view_errors = reprojection_error(world_points, selected_corners, mtx, dist, rvecs, tvecs)
    print("Reprojection Error: ", re)
    print(mtx)
    print(dist)

    # mtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    np.savez('{}/camera_calibration_{}'.format(intrinsics_dir, cam_id), RMSE=ret, calibration_mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    corners_3d = [project_world_to_camera(world_points[i], rvecs[i], tvecs[i]) for i in range(len(world_points))]

    thresholder = Thresholder(opts.error_threshold)
    if not opts.headless:
        visualize_errors(thresholder, cam_id, per_view_errors)
    
    per_view_refined = per_view_errors[per_view_errors < thresholder.threshold]
    indexes = np.array([np.argwhere(per_view_errors == a) for a in per_view_refined])
    indexes = indexes.reshape(-1)
    print("Indexes below threshold: ", len(indexes))
    if len(indexes)  < 10:
        print("Too little sample size, consider increasing the refine param")
        return ret, corners_3d
    world_points = np.array(world_points)
    selected_corners = np.array(selected_corners)
    world_refined_points = world_points[indexes, :]
    selected_corners_refined = selected_corners[indexes, :]
    
    refined_ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_refined_points, selected_corners_refined, (width, height), None, None, flags=opts.flags, criteria=opts.criteria)
    
    if not opts.headless:
        visualize_errors(thresholder, cam_id, per_view_refined)
    
    print("{} - Refined RMSE: ".format(cam_id), refined_ret)
    print(mtx)
    print(dist)
    if refined_ret < ret:
        ret = refined_ret
        # mtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
        np.savez('{}/camera_calibration_{}'.format(intrinsics_dir, cam_id), RMSE=refined_ret, calibration_mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        corners_3d = [project_world_to_camera(world_refined_points[i], rvecs[i], tvecs[i]) for i in range(len(world_refined_points))]
    return ret, corners_3d


def single_camera_rectification(opts: SingleCameraCalibrateOptions) -> None:
    if opts.headless:
        return
    intrinsic_params = np.load(Path(opts.dir).absolute().joinpath("camera_calibration_{}.npz".format(opts.cam_id)))
    mtx, dist = intrinsic_params["calibration_mtx"], intrinsic_params["dist"]

    images = opts.load_single_images()
    count = len(images)
    index = 0
    w_name = "Undistorted Image {}".format(opts.cam_id)
    opts.cv_options.named_window(w_name)
    frame = cv.imread(images[index])
    dst = cv.undistort(frame, mtx, dist, mtx)
    dst = cv.putText(dst, "Index {} | Press space to view next image; q to quit".format(index),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
    while True:
        key = cv.waitKey(0) & 0xFF
        cv.imshow(w_name, dst)
        if key == ord(' '):
            index += 1
            if count <= index:
                break
            frame = cv.imread(images[index])
            dst = cv.undistort(frame, mtx, dist, mtx)
            dst = cv.putText(dst, "Index {} | Press space to view next image; q to quit".format(index),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
        
        if key == ord('q'):
            break
    cv.destroyAllWindows()
            
def reprojection_error(world_points, corners, mtx, dist, rvecs, tvecs):
    se = 0
    total_count = len(world_points) * world_points[0].shape[0]
    per_view_count = world_points[0].shape[0]
    per_view_error = list()
    for i in range(len(world_points)):
        reprojected_points, _ = cv.projectPoints(world_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = np.linalg.norm((corners[i] - reprojected_points)) ** 2
        per_view_error.append(np.sqrt(error / per_view_count))
        se += error
    mse = se / total_count
    return np.sqrt(mse), np.array(per_view_error)

def project_axis(world_points, corners, mtx, dist):
    axis = np.array([[[3.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, -6.0]]], dtype=np.float32)
    ret, rvec, tvec = cv.solvePnP(world_points, corners, mtx, dist)
    reprojected_points, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
    return reprojected_points

