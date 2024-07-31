import numpy as np
import cv2 as cv
from typing import *
from pathlib import Path
import jsonpickle

from options import DepthEsitmationOptions

def create_trackbar(name):
    nothing = lambda x: ()
    cv.namedWindow(name,cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 600,600)
    
    cv.createTrackbar('numDisparities', name ,1, 50,nothing)
    cv.createTrackbar('blockSize',name,5,50,nothing)
    cv.createTrackbar('preFilterType',name,0,1,nothing)
    cv.createTrackbar('preFilterSize',name,2,25,nothing)
    cv.createTrackbar('preFilterCap',name,5,62,nothing)
    cv.createTrackbar('textureThreshold',name,10,100,nothing)
    cv.createTrackbar('uniquenessRatio',name,15,100,nothing)
    cv.createTrackbar('speckleRange',name,0,100,nothing)
    cv.createTrackbar('speckleWindowSize',name,3,25,nothing)
    cv.createTrackbar('disp12MaxDiff',name,5,25,nothing)
    cv.createTrackbar('minDisparity',name,5,25,nothing)
    

def get_stereo_depth(opts: DepthEsitmationOptions):
    extrinsics_dir = ""
    left_map_x, left_map_y, right_map_x, right_map_y = opts.load_remaps()
    images = opts.load_paired_images()
    index = 0

    stereo_name = "stereo disparity map params"
    sgbm_name = "sgbm disparity map params"
    create_trackbar(stereo_name)
    create_trackbar(sgbm_name)
    
    ret, stereo = load_stereo_object("random.json")
    if not ret:
        stereo: cv.StereoBM = cv.StereoBM_create()

    ret, stereoSGBM = load_stereo_object("stereo_sgbm_disparity2.json")
    if not ret:
        stereoSGBM: cv.StereoSGBM = cv.StereoSGBM_create()
    minDisparity, numDisparities = 5, 16
    left_frame = cv.imread(images[index][0])
    right_frame = cv.imread(images[index][1])
    left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
    right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    
    norm_stereo = lambda x: (x - stereo.getMinDisparity()) / stereo.getNumDisparities()
    norm_sgbm_stereo = lambda x: cv.normalize(x, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    stereo_disparity = get_stereo_disparity(stereo, index, left_frame_rect, right_frame_rect, norm_sgbm_stereo)
    SGBM_disparity = get_stereo_disparity(stereoSGBM, index, left_frame_rect, right_frame_rect, norm_sgbm_stereo)
    cv.imshow("SGBM Disparity", SGBM_disparity)
    cv.imshow("Stereo Disparity", stereo_disparity)

    while True:
        key = cv.waitKey(1)
        if key & 0xFF == ord(' '):
            print("next image...")
            left_frame = cv.imread(images[index][0])
            right_frame = cv.imread(images[index][1])
            left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
            right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)

            stereo_disparity = get_stereo_disparity(stereo, index, left_frame_rect, right_frame_rect, norm_stereo)
            SGBM_disparity = get_stereo_disparity(stereoSGBM, index, left_frame_rect, right_frame_rect, norm_sgbm_stereo)
            cv.imshow("SGBM Disparity", SGBM_disparity)
            cv.imshow("Stereo Disparity", stereo_disparity)
            index += 1
            
        
        if key & 0xFF == ord('c'): 
            stereo = update_stereo(stereo_name, stereo)
            stereoSGBM = update_stere_sgbm(sgbm_name, stereoSGBM)
            stereo_disparity = get_stereo_disparity(stereo, index, left_frame_rect, right_frame_rect, norm_stereo)
            SGBM_disparity = get_stereo_disparity(stereoSGBM, index, left_frame_rect, right_frame_rect, norm_sgbm_stereo)
            cv.imshow("SGBM Disparity", SGBM_disparity)
            cv.imshow("Stereo Disparity", stereo_disparity)

        
        if key & 0xFF == ord('q'):
            break

        if index >= opts.count:
            break

    cv.destroyAllWindows()
    write_stereo_object(stereo)
    write_stereo_object(stereoSGBM, file_name="stereo_sgbm_disparity.json")
    

def load_stereo_object(file_name) -> list[bool, cv.StereoBM]:
    try:
        with open(file_name, "rb") as f:
            data = f.read()
            if not data:
                False, None
            return True, jsonpickle.decode(data)
    except:
        return False, None


def write_stereo_object(obj: cv.StereoSGBM | cv.StereoBM, file_name="stereo_disparity.json"):
    data = jsonpickle.encode(obj)
    with open(file_name, 'w') as file:
        file.write(data)
        file.close()

def get_stereo_disparity(stereo: cv.StereoBM | cv.StereoSGBM, index:int, left_frame_rect: np.ndarray, right_frame_rect:np.ndarray, normalize: Callable) -> np.ndarray:
    disparity = stereo.compute(cv.cvtColor(left_frame_rect, cv.COLOR_BGR2GRAY), cv.cvtColor(right_frame_rect, cv.COLOR_BGR2GRAY))
    disparity = normalize(disparity)
    # disparity = cv.applyColorMap(disparity, cv.COLORMAP_JET)
    disparity_text = cv.putText(disparity, "Index: {}. Press space to move to the next image. press q to quit".format(index),(100, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, 1)
    return disparity_text

def update_stereo(name:str, stereo: cv.StereoBM):
    numDisparities = (cv.getTrackbarPos('numDisparities',name))*16
    blockSize = (cv.getTrackbarPos('blockSize',name))*2 + 5
    preFilterType = cv.getTrackbarPos('preFilterType',name)
    preFilterSize = cv.getTrackbarPos('preFilterSize',name)*2 + 5
    preFilterCap = cv.getTrackbarPos('preFilterCap',name)
    textureThreshold = cv.getTrackbarPos('textureThreshold',name)
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio',name)
    speckleRange = cv.getTrackbarPos('speckleRange',name)
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize',name)*2
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff',name)
    minDisparity = cv.getTrackbarPos('minDisparity',name)
    P1 = 8 * 3 * blockSize ** 2
    P2 = 32 * 3 * blockSize ** 2

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
    return stereo

def update_stere_sgbm(name:str, stereo: cv.StereoSGBM):
    numDisparities = (cv.getTrackbarPos('numDisparities',name))*16
    blockSize = (cv.getTrackbarPos('blockSize',name))*2 + 5
    preFilterCap = cv.getTrackbarPos('preFilterCap',name)
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio',name)
    speckleRange = cv.getTrackbarPos('speckleRange',name)
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize',name)*2
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff',name)
    minDisparity = cv.getTrackbarPos('minDisparity',name)
    P1 = 8 * 3 * blockSize ** 2
    P2 = 32 * 3 * blockSize ** 2

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setP1(P1)
    stereo.setP2(P2)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    return stereo
