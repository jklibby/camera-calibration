import numpy as np
import cv2 as cv
import json
import os
from utils import checkerboard_orientation
from stereo_distance import get_3d_points

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

    if not (left_cap.isOpened() and right_cap.isOpened()):
        print("Cannot read video frames")
        exit()
    
    poi = POICollection(window="Depth")

    cv.namedWindow("Left Frame | Disparity | Depth Map")
    cv.setMouseCallback("Left Frame | Disparity | Depth Map", poi.add)
    
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
        
        if hold:
            cv.imshow("Left Frame | Disparity | Depth Map", depth)
            if len(poi.collection) > 0:
                print(depth[poi.collection[-1]])
            continue
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

        lret_corners, lcorners = cv.findChessboardCorners(left_frame, (10, 7))
        rret_corners, rcorners = cv.findChessboardCorners(right_frame, (10, 7))
        text_left_frame_rect = cv.putText(left_frame_rect.copy(), "{} - {}".format(lret_corners, rret_corners), (100, 100), cv.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        if lret_corners and rret_corners:
            ldir, rdir = checkerboard_orientation(lcorners), checkerboard_orientation(rcorners)
            if ldir == rdir:
                print(ldir)
                p1, p2 = lcorners[0:3], rcorners[0:3]
                pcd = get_3d_points(p1, p2, dir)
                world_scaling = 1.5
                return 2*world_scaling - np.linalg.norm(pcd[0] - pcd[-1]) * world_scaling# test grid dimension
                
        cv.imshow("Recitfied Images", np.hstack([text_left_frame_rect, right_frame_rect]))

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