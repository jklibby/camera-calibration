import numpy as np
import cv2 as cv
from typing import *
from pathlib import Path
from datetime import datetime

from options import DepthEstimationOptions

def create_trackbar(name:str, stereo: cv.StereoBM):
    nothing = lambda x: ()
    cv.namedWindow(name,cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 600,600)
    cv.createTrackbar('numDisparities', name,int(stereo.getNumDisparities() / 16), 50,nothing)
    cv.createTrackbar('blockSize',name,int((stereo.getBlockSize() - 5) / 2),50,nothing)
    cv.createTrackbar('preFilterType',name,stereo.getPreFilterType(),1,nothing)
    cv.createTrackbar('preFilterSize',name,int((stereo.getBlockSize() - 5) / 2),50,nothing)
    cv.createTrackbar('preFilterCap',name,stereo.getPreFilterCap(),50,nothing)
    cv.createTrackbar('textureThreshold',name,stereo.getTextureThreshold(),100,nothing)
    cv.createTrackbar('uniquenessRatio',name,stereo.getUniquenessRatio(),100,nothing)
    cv.createTrackbar('speckleRange',name,stereo.getSpeckleRange(),100,nothing)
    cv.createTrackbar('speckleWindowSize',name,int(stereo.getSpeckleWindowSize() / 2),50,nothing)
    cv.createTrackbar('disp12MaxDiff',name,max(stereo.getDisp12MaxDiff(), 0),10,nothing)
    cv.createTrackbar('minDisparity',name,stereo.getMinDisparity(), 0,nothing)

def create_trackbar_sgbm(name, stereo_sgbm: cv.StereoSGBM):
    nothing = lambda x: ()
    cv.namedWindow(name,cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 600,600)
    
    cv.createTrackbar('numDisparities', name, stereo_sgbm.getNumDisparities(), 50,nothing)
    cv.createTrackbar('blockSize',name, stereo_sgbm.getBlockSize(),50,nothing)
    cv.createTrackbar('P1',name, stereo_sgbm.getP1(),10,nothing)
    cv.createTrackbar('P2',name, stereo_sgbm.getP2(),50,nothing)
    cv.createTrackbar('preFilterCap',name, stereo_sgbm.getPreFilterCap(),62,nothing)
    cv.createTrackbar('uniquenessRatio',name, stereo_sgbm.getUniquenessRatio(),100,nothing)
    cv.createTrackbar('speckleRange',name, stereo_sgbm.getSpeckleRange(),100,nothing)
    cv.createTrackbar('speckleWindowSize',name, stereo_sgbm.getSpeckleWindowSize(),25,nothing)
    cv.createTrackbar('disp12MaxDiff',name, max(stereo_sgbm.getDisp12MaxDiff(), 0),25,nothing)
    cv.createTrackbar('minDisparity',name, stereo_sgbm.getMinDisparity(),25,nothing)
    

def get_stereo_depth(opts: DepthEstimationOptions) -> None:
    """
        This function provides a GUI to creating and tuning params for 
        Block Matching and Semi Global Block Matching Depth Map generation. 

        Outputs:
            BM params: params stored in OpenCV formatted yaml file in 
                opts.extrinsic_dir/depth_estimation/stereo_bm.yaml
            Semi Global BM params: params stored in OpenCV formatted yaml file in 
                opts.extrinsic_dir/depth_estimation/stereo_sgbm.yaml
    """
    extrinsics_dir = ""
    left_map_x, left_map_y, right_map_x, right_map_y = opts.load_remaps()
    images = opts.load_paired_images()
    index = 0

    ret, stereo = load_stereo_object("stereo_bm_v1.yaml")
    if not ret:
        stereo: cv.StereoBM = cv.StereoBM_create()

    ret, stereoSGBM = load_stereo_sgbm_object("stereo_sgbm_v1.yaml")
    if not ret:
        stereoSGBM: cv.StereoSGBM = cv.StereoSGBM_create()
    
    stereo_name = "stereo disparity map params"
    sgbm_name = "sgbm disparity map params"
    create_trackbar(stereo_name, stereo)
    create_trackbar_sgbm(sgbm_name, stereoSGBM)
    

    left_frame = cv.imread(images[index][0])
    right_frame = cv.imread(images[index][1])
    left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
    right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    
    norm_stereo = lambda s, x: cv.normalize((x/16.0 - (s.getMinDisparity() / s.getNumDisparities())), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    norm_sgbm_stereo = lambda stereo, x: cv.normalize(x, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    stereo_disparity = get_stereo_disparity(stereo, left_frame_rect, right_frame_rect, norm_stereo)
    SGBM_disparity = get_stereo_disparity(stereoSGBM, left_frame_rect, right_frame_rect, norm_stereo)
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
            
            stereo_disparity = get_stereo_disparity(stereo, left_frame_rect, right_frame_rect, norm_stereo)
            SGBM_disparity = get_stereo_disparity(stereoSGBM, left_frame_rect, right_frame_rect, norm_sgbm_stereo)
            cv.imshow("SGBM Disparity", SGBM_disparity)
            cv.imshow("Stereo Disparity", stereo_disparity)
            index += 1
            
        
        if key & 0xFF == ord('c'): 
            stereo = update_stereo(stereo_name, stereo)
            stereoSGBM = update_stere_sgbm(sgbm_name, stereoSGBM)
            stereo_disparity = get_stereo_disparity(stereo, left_frame_rect, right_frame_rect, norm_stereo)
            SGBM_disparity = get_stereo_disparity(stereoSGBM, left_frame_rect, right_frame_rect, norm_sgbm_stereo)

            cv.imshow("SGBM Disparity", SGBM_disparity)
            cv.imshow("Stereo Disparity", stereo_disparity)
        
        if key & 0xFF == ord('w'):
            wls_filter_disparity = wlsfilter_disparity(left_frame_rect, right_frame_rect, stereo_disparity, stereo)
            sgbm_wls_filter_disparity = wlsfilter_disparity(left_frame_rect, right_frame_rect, SGBM_disparity, stereoSGBM)
            cv.imshow("SGBM Disparity", sgbm_wls_filter_disparity)
            cv.imshow("Stereo Disparity", wls_filter_disparity)


        
        if key & 0xFF == ord('q'):
            break

        if index >= opts.count:
            break

    cv.destroyAllWindows()
    extrinsics_dir = Path(opts.extrinsics_dir)
    bm_path = str(extrinsics_dir.joinpath(["depth_estimation", "stereo_bm.yaml"]))
    sgbm_path = str(extrinsics_dir.joinpath(["depth_estimation", "stereo_sgbm.yaml"]))
    write_stereo_object(stereo, file_name=bm_path)
    write_stereo_object(stereoSGBM, file_name=sgbm_path)
    

def get_live_stereo_depth(opts: DepthEstimationOptions):
    left_map_x, left_map_y, right_map_x, right_map_y = opts.load_remaps()
    images = opts.load_paired_images()
    index = 0

    stereo_name = "stereo disparity map params"
    sgbm_name = "sgbm disparity map params"
    create_trackbar(stereo_name)
    create_trackbar_sgbm(sgbm_name)
    
    ret, stereo = load_stereo_object("random.json")
    if not ret:
        stereo: cv.StereoBM = cv.StereoBM_create()

    ret, stereoSGBM = load_stereo_object("stereo_sgbm_disparity2.json")
    if not ret:
        stereoSGBM: cv.StereoSGBM = cv.StereoSGBM_create()
    
    opts.cv_options.named_window("Left Camera")
    opts.cv_options.named_window("Right Camera")
    opts.cv_options.named_window("Stereo BM Depth")
    opts.cv_options.named_window("StereoSGBM Depth")

    left_cap = cv.VideoCapture(opts.left_cam_id)
    right_cap = cv.VideoCapture(opts.right_cam_id)


    if (not left_cap.isOpened()) or (not right_cap.isOpened):
        raise Exception("Error while start live stream")
    
    norm_stereo = lambda x: (x - stereo.getMinDisparity()) / stereo.getNumDisparities()
    norm_sgbm_stereo = lambda x: cv.normalize(x, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    while True:
        lret, left_frame = left_cap.read()
        rret, right_frame = right_cap.read()
        if not (lret or rret):
            break
        left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
        right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)

        stereo_disparity = get_stereo_disparity(stereo, index, left_frame_rect, right_frame_rect, norm_sgbm_stereo)
        SGBM_disparity = get_stereo_disparity(stereoSGBM, index, left_frame_rect, right_frame_rect, norm_sgbm_stereo)
        cv.imshow("Left Camera", left_frame_rect)
        cv.imshow("Right Camera", right_frame_rect)
        cv.imshow("Stereo BM Depth", stereo_disparity)
        cv.imshow("SGBM_disparity", SGBM_disparity)
    
    cv.destroyAllWindows()
    left_cap.release()
    right_cap.release()

def load_stereo_object(file_name) -> list[bool, cv.StereoBM]:
    try:
        fs = cv.FileStorage(file_name, cv.FILE_STORAGE_READ)
        sbm: cv.StereoBM = cv.StereoBM_create()
        sbm.read(fs.getFirstTopLevelNode())
        return True, sbm
    except Exception as e:
        print("err", e)
        return False, None
    
def load_stereo_sgbm_object(file_name) -> list[bool, cv.StereoSGBM]:
    try:
        fs = cv.FileStorage(file_name, cv.FILE_STORAGE_READ)
        sbm: cv.StereoSGBM = cv.StereoSGBM_create()
        sbm.read(fs.getFirstTopLevelNode())
        return True, sbm
    except Exception as e: 
        print("err", e)
        return False, None


def write_stereo_object(obj: cv.StereoSGBM | cv.StereoBM, file_name:str):
    obj.save(file_name)

def get_stereo_disparity(stereo: cv.StereoBM | cv.StereoSGBM, left_frame_rect: np.ndarray, right_frame_rect:np.ndarray, normalize: Callable) -> np.ndarray:
    start = datetime.now()
    left_frame_rect, right_frame_rect = pad_images(stereo.getNumDisparities(), left_frame_rect, right_frame_rect)
    disparity = stereo.compute(left_frame_rect, right_frame_rect)
    disparity = unpad_disparity(stereo.getNumDisparities(), disparity)
    disparity = normalize(stereo, disparity)
    print("FPS...", stereo, 1/(datetime.now()-start).total_seconds())
    return disparity

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
    preFilterCap = cv.getTrackbarPos('preFilterCap', name)
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio',name)
    speckleRange = cv.getTrackbarPos('speckleRange',name)
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize',name)*2
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff',name)
    minDisparity = cv.getTrackbarPos('minDisparity',name)
    P1 = cv.getTrackbarPos('P1', name)
    P2 = cv.getTrackbarPos('P2', name)
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

def pad_images(num_disparity:int, left_rect_image:np.ndarray, right_rect_image:np.ndarray):
    left_rect_image, right_rect_image = cv.cvtColor(left_rect_image, cv.COLOR_BGR2GRAY), cv.cvtColor(right_rect_image, cv.COLOR_BGR2GRAY)
    height = left_rect_image.shape[0]
    padding = np.zeros((height, num_disparity), dtype=np.uint8)
    left_rect_image = np.hstack([padding, left_rect_image])
    right_rect_image = np.hstack([padding, right_rect_image])
    return left_rect_image, right_rect_image

def unpad_disparity(numDisparity:int, disparity: np.ndarray):
    return disparity[:, numDisparity:]

def wlsfilter_disparity(left_image:np.ndarray, right_image: np.ndarray, disparity: np.ndarray, stereo:cv.StereoBM):
    disparity.dtype = np.uint8
    start = datetime.now()
    sigma = 1.5
    lmbda = 8000.0
    right_matcher = cv.ximgproc.createRightMatcher(stereo)
    left_image, right_image = pad_images(right_matcher.getNumDisparities(), left_image, right_image)
    right_disp = right_matcher.compute(right_image,left_image)
    right_disp = unpad_disparity(right_matcher.getNumDisparities(), right_disp)

    wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(disparity, left_image, disparity_map_right=right_disp)
    print("WLS...",stereo, (datetime.now()-start).total_seconds())
    return filtered_disp