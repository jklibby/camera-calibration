import numpy as np
import cv2 as cv
import json
import time
from depth_estimation.stereo_distance import get_3d_points
from utils import get_CB_corners

def get_stereo_depth(count, path):
    remaps = np.load('stereo-rectified-maps.npz')
    left_map_x, left_map_y, right_map_x, right_map_y = remaps['left_map_x'], remaps['left_map_y'], remaps['right_map_x'], remaps['right_map_y']
    frame_pair_dir = path
    images = [('{}/camera0_{}.png'.format(frame_pair_dir, cap_count), '{}/camera1_{}.png'.format(frame_pair_dir, cap_count)) for cap_count in range(count)]
    index = 0

    def nothing(x):
       pass
 
    cv.namedWindow('disp',cv.WINDOW_NORMAL)
    cv.resizeWindow('disp',600,600)
    
    cv.createTrackbar('numDisparities','disp',1,17,nothing)
    cv.createTrackbar('blockSize','disp',5,50,nothing)
    cv.createTrackbar('preFilterType','disp',0,1,nothing)
    cv.createTrackbar('preFilterSize','disp',2,25,nothing)
    cv.createTrackbar('preFilterCap','disp',5,62,nothing)
    cv.createTrackbar('textureThreshold','disp',10,100,nothing)
    cv.createTrackbar('uniquenessRatio','disp',15,100,nothing)
    cv.createTrackbar('speckleRange','disp',0,100,nothing)
    cv.createTrackbar('speckleWindowSize','disp',3,25,nothing)
    cv.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
    cv.createTrackbar('minDisparity','disp',5,25,nothing)
    ret, stereo = create_stereo_object()
    if not ret:
        stereo: cv.StereoBM = cv.StereoBM_create()

    minDisparity, numDisparities = 5, 16
    left_frame = cv.imread(images[index][0])
    right_frame = cv.imread(images[index][1])
    left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
    right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    while True:
        disparity = stereo.compute(cv.cvtColor(left_frame_rect, cv.COLOR_BGR2GRAY), cv.cvtColor(right_frame_rect, cv.COLOR_BGR2GRAY))
        # disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # display depth map for each image
        disparity_text = cv.putText(disparity, "Index: {}. Press space to move to the next image. press q to quit".format(index),(100, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, 1)
        # cv.imshow("Left Rectified image", left_frame_rect)
        cv.imshow("Disparity", disparity_text)
        
        key = cv.waitKey(1)
        if key & 0xFF == ord(' '):
            print("next image...")
            left_frame = cv.imread(images[index][0])
            right_frame = cv.imread(images[index][1])
            left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
            right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
            index += 1
        
        if key & 0xFF == ord('c'):
            
            numDisparities = (cv.getTrackbarPos('numDisparities','disp'))*16
            blockSize = (cv.getTrackbarPos('blockSize','disp'))*2 + 5
            preFilterType = cv.getTrackbarPos('preFilterType','disp')
            preFilterSize = cv.getTrackbarPos('preFilterSize','disp')*2 + 5
            preFilterCap = cv.getTrackbarPos('preFilterCap','disp')
            textureThreshold = cv.getTrackbarPos('textureThreshold','disp')
            uniquenessRatio = cv.getTrackbarPos('uniquenessRatio','disp')
            speckleRange = cv.getTrackbarPos('speckleRange','disp')
            speckleWindowSize = cv.getTrackbarPos('speckleWindowSize','disp')*2
            disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff','disp')
            minDisparity = cv.getTrackbarPos('minDisparity','disp')
            
            # Setting the updated parameters before computing disparity map
            stereo.setNumDisparities(numDisparities)
            stereo.setBlockSize(blockSize)
            stereo.setPreFilterType(preFilterType)
            stereo.setPreFilterSize(preFilterSize)
            stereo.setPreFilterCap(preFilterCap)
            stereo.setTextureThreshold(textureThreshold)
            stereo.setUniquenessRatio(uniquenessRatio)
            stereo.setSpeckleRange(speckleRange)
            stereo.setSpeckleWindowSize(speckleWindowSize)
            stereo.setDisp12MaxDiff(disp12MaxDiff)
            stereo.setMinDisparity(minDisparity)
        
        if key & 0xFF == ord('q'):
            break

        if index >= len(images):
            break

    cv.destroyAllWindows()

    file = open('stereo_disparity.json', 'wt')
    stereo_save = dict()
    stereo_save["numDisparities"] = stereo.getNumDisparities()
    stereo_save["blockSize"] = stereo.getBlockSize()
    stereo_save["preFilterType"] = stereo.getPreFilterType()
    stereo_save["preFilterSize"] = stereo.getPreFilterSize()
    stereo_save["preFilterCap"] = stereo.getPreFilterCap()
    stereo_save["speckleRange"] = stereo.getSpeckleRange()
    stereo_save["uniquenessRatio"] = stereo.getUniquenessRatio()
    stereo_save["textureThreshold"] = stereo.getTextureThreshold()
    stereo_save["speckleRange"] = stereo.getSpeckleRange()
    stereo_save["speckleWindowSize"] = stereo.getSpeckleWindowSize()
    stereo_save["disp12MaxDiff"] = stereo.getDisp12MaxDiff()
    stereo_save["minDisparity"] = stereo.getMinDisparity()
    json.dump(stereo_save, file)

# TO-DO: rewrite
def live_stereo_depth(left_cam, right_cam, baseline=25/1000):
    ret, stereo = create_stereo_object()
    if not ret:
        return

    intrinsic_0 = np.load("intrinsics/camera_calibration-0.npz")
    K = intrinsic_0["calibration_mtx"]

    remaps = np.load('stereo-rectified-maps.npz')
    left_map_x, left_map_y, right_map_x, right_map_y = remaps['left_map_x'], remaps['left_map_y'], remaps['right_map_x'], remaps['right_map_y']
    
    left_cap = cv.VideoCapture(left_cam)
    right_cap = cv.VideoCapture(right_cam)

    #TO-DO: Load from config
    left_cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    left_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    right_cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    right_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)


    if not (left_cap.isOpened() and right_cap.isOpened()):
        print("Cannot read video frames")
        exit()
    

    cv.namedWindow("Left Frame | Disparity | Depth Map")
    
    hold = False
    while True:
        lret, left_frame = left_cap.read()
        rret, right_frame = right_cap.read()
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            left_cap.release()
            right_cap.release()
            cv.destroyAllWindows()
            break
        
        if key == ord('h'):
            hold = not hold
            print(hold)
        
        if not (rret and lret):
            print("Cannot read video frames")
            left_cap.release()
            right_cap.release()
            break
        
        left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
        right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
        left_frame_rect = cv.cvtColor(left_frame_rect, cv.COLOR_BGR2GRAY)
        right_frame_rect = cv.cvtColor(right_frame_rect, cv.COLOR_BGR2GRAY)
        disparity = stereo.compute(left_frame_rect, right_frame_rect)
        disparity = (disparity / 16.0 - stereo.getMinDisparity()) / stereo.getNumDisparities()
        disparity[disparity == 0] = 1e-2
        depth: np.ndarray  = (K[1, 1] * baseline) / disparity
        cv.imshow("Left Frame | Disparity | Depth Map", depth)
        


def create_stereo_object() -> list[bool, cv.StereoBM]:
    stereo_options = json.load(open("stereo_disparity.json", "rb"))
    if not stereo_options:
        False, None
    stereo = cv.StereoBM_create()

    numDisparities = stereo_options["numDisparities"]
    blockSize = stereo_options["blockSize"]
    preFilterType = stereo_options["preFilterType"]
    preFilterSize = stereo_options["preFilterSize"] 
    preFilterCap = stereo_options["preFilterCap"]
    speckleRange = stereo_options["speckleRange"]
    uniquenessRatio = stereo_options["uniquenessRatio"]
    textureThreshold = stereo_options["textureThreshold"]
    speckleWindowSize = stereo_options["speckleWindowSize"]
    disp12MaxDiff = stereo_options["disp12MaxDiff"]
    minDisparity = stereo_options["minDisparity"]

    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    return True, stereo


def get_object_measurement(left_cam, right_cam, pattern, dir):
    remaps = np.load('extrinsics/{}/stereo-rectified-maps.npz'.format(dir))
    left_map_x, left_map_y, right_map_x, right_map_y = remaps['left_map_x'], remaps['left_map_y'], remaps['right_map_x'], remaps['right_map_y']
    
    left_cap = cv.VideoCapture(left_cam)
    right_cap = cv.VideoCapture(right_cam)
    video_err = not (left_cap.isOpened() and right_cap.isOpened())
    if video_err:
        print("Cannot read video frames {}".format(video_err))
        print(left_cap.isOpened(), right_cap.isOpened(), left_cap.isOpened() and right_cap.isOpened())
        exit()
    
    index = 0
    while True:
        lret, left_frame = left_cap.read()
        rret, right_frame = right_cap.read()
        key = cv.waitKey(1) & 0xFF

        left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
        right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)

        lret_corners, lcorners = cv.findChessboardCorners(left_frame_rect, pattern)
        rret_corners, rcorners = cv.findChessboardCorners(right_frame_rect, pattern)
        text_left_frame_rect = cv.putText(left_frame_rect.copy(), "{} - {}".format(lret_corners, rret_corners), (100, 100), cv.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        if lret_corners and rret_corners:
            p1, p2 = lcorners, rcorners
            pcd = get_3d_points(p1, p2, dir)
            cv.imwrite("validation_cb_0.png", left_frame_rect)
            cv.imwrite("validation_cb_1.png", right_frame_rect)
            world_scaling = 1.5
            # TO-DO - Load world scaling or calculate it?
            np.save("3d_corners_checkerboard".format(time.asctime()), pcd)
            return np.linalg.norm(pcd[0] - pcd[-1]) * world_scaling# test grid dimension
                
        cv.imshow("Left Recitfied Images", left_frame_rect)
        cv.imshow("Right Recitfied Images", right_frame_rect)

        if key == ord('q'):
            left_cap.release()
            right_cap.release()
            cv.destroyAllWindows()
            break
        
        if not (rret and lret):
            print("Cannot read video frames")
            left_cap.release()
            right_cap.release()
            break
    return 0


def get_checkerboard_3d(count, pattern, dir):
    remaps = np.load('extrinsics/{}/stereo-rectified-maps.npz'.format(dir))
    left_map_x, left_map_y, right_map_x, right_map_y = remaps['left_map_x'], remaps['left_map_y'], remaps['right_map_x'], remaps['right_map_y']
    
    # load paired images
    images = [('paired_images/camera0_{}.png'.format(cap_count), 'paired_images/camera1_{}.png'.format(cap_count)) for cap_count in range(count)]
    corners = list()
    for li, ri in images:
        left_frame = cv.imread(li)
        right_frame = cv.imread(ri)

        # apply rectification
        left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
        right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)

        # detect checkerboards
        lret, _, left_corner = get_CB_corners(left_frame_rect, pattern)
        rret, _, right_corner = get_CB_corners(right_frame_rect, pattern)
        if lret and rret:
            corner_pcd = get_3d_points(left_corner, right_corner, dir)
            corners.append(corner_pcd)
    
    return np.array(corners)