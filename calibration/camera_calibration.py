import numpy as np
import cv2 as cv
import os

from utils.calibration import get_CB_corners

def single_camera_calibrate(cam_id, count, path, pattern_size=(7, 7), flags=None, dir=None, skip_gui=False, refine_val=1):
    # read all the images
    intrinsics_dir = 'intrinsics'
    if dir:
        intrinsics_dir = 'intrinsics/{}'.format(dir)
    if not os.path.exists(intrinsics_dir):
        os.makedirs(intrinsics_dir)
    dir = '{}/{}'.format(path, cam_id)
    if not os.path.exists(dir):
        print("Directory does not exists, capture images first")
        return
    

    wp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    wp[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)

    images = ['{}/camera{}_{}.png'.format(dir, cam_id, cap_count) for cap_count in range(count)]
    world_points = list()
    height, width = 0, 0
    display_frames = list()
    for idx, image in enumerate(images):
        frame = cv.imread(image)
        height = frame.shape[0]
        width = frame.shape[1]
        g_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # find chessboard corners
        ret, pattern_frame, corners = get_CB_corners(frame, pattern_size)
        if ret:
            start_frame = cv.putText(pattern_frame, "Press space to find chessboard corners; s to skip to skip current frame; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
            display_frames.append((start_frame, corners))


    selected_corners = list()
    curr_count = count
    index = 0
    while True:
        if curr_count <= 0 or index >= len(display_frames):
            break
        if skip_gui:
            cv.imshow("Frames {}".format(cam_id), display_frames[index][0])
            key = cv.waitKey(1)
            # if key & 0xFF == ord(' '):
            selected_corners.append(display_frames[index][1])
            world_points.append(wp)
            index += 1
            curr_count -= 1

            if key & 0xFF == ord('s'):
                print('skipped: {}'.format(index))
                index += 1
        
            if key & 0xFF == ord('q'):
                break
        else:
            selected_corners.append(display_frames[index][1])
            world_points.append(wp)
            index += 1
            curr_count -= 1
        
            
        
    cv.destroyAllWindows()
    # calculate intrinsic params
    if flags:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_points, selected_corners, (width, height), None, None, flags=flags)
    else:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_points, selected_corners, (width, height), None, None)

    print("{} - RMSE: ".format(cam_id), ret)
    re, per_view_errors = reprojection_error(world_points, selected_corners, mtx, dist, rvecs, tvecs)
    print("Reprojection Error: ", re)
    print(mtx)
    print(dist)

    np.savez('{}/camera_calibration_{}'.format(intrinsics_dir, cam_id), calibration_mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    if refine_val < 0:
        return ret
    per_view_refined = per_view_errors[per_view_errors < refine_val]
    indexes = np.array([np.argwhere(per_view_errors == a) for a in per_view_refined])
    indexes = indexes.reshape(-1)
    print("Indexes below threshold: ", len(indexes))
    if len(indexes)  < 3:
        print("Too little sample size, consider increasing the refine param")
        return ret
    world_points = np.array(world_points)
    selected_corners = np.array(selected_corners)
    world_refined_points = world_points[indexes, :]
    selected_corners_refined = selected_corners[indexes, :]
    if indexes.shape[0] < 15:
        print("Too few refined images")
        return ret
    if flags:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_refined_points, selected_corners_refined, (width, height), None, None, flags=flags)
    else:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_refined_points, selected_corners_refined, (width, height), None, None)

    print("{} - Refined RMSE: ".format(cam_id), ret)
    print(mtx)
    print(dist)
    np.savez('{}/camera_calibration_{}'.format(intrinsics_dir, cam_id), calibration_mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    return ret


def single_camera_pose_estimation(cam_id, pattern_size=(7, 7), dir=None):
    cc = np.load('intrinsics/{}/camera_calibration_{}.npz'.format(cam_id))
    mtx, dist = cc['calibration_mtx'], cc['dist']
    cap = cv.VideoCapture(cam_id)

    #TO-DO: Config-file
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    wp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    wp[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)
    while True:
        ret, frame = cap.read()
        key = cv.waitKey(1)
        
        if not ret:
            print("Cannot read video frames")
            cap.released()
            break
        if key & 0xFF == ord('q'):
            break

        # detect chessboard
        ret, _, corners = get_CB_corners(frame, pattern_size)
        if ret:
            projected_axis = project_axis(wp, corners, mtx, dist)
            pose_image = draw_pose(frame, corners[0], projected_axis)
            cv.imshow("Pose", pose_image)
        else:
            cv.imshow("Pose", frame)
    cv.destroyAllWindows()
    cap.release()

            
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

def draw_pose(img, corner, reprojected_axis):
    reprojected_axis = reprojected_axis.reshape(-1, 2)
    reprojected_axis = reprojected_axis.astype(np.int32)
    corner = corner.flatten()
    corner = corner.astype(np.int32)
    cv.line(img, corner, reprojected_axis[0], (255, 0, 0), thickness=3, lineType=cv.LINE_4)
    cv.line(img, corner, reprojected_axis[1], (0, 255, 0), thickness=3, lineType=cv.LINE_4)
    cv.line(img, corner, reprojected_axis[2], (0, 0, 255), thickness=3, lineType=cv.LINE_4)
    return img
