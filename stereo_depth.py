import numpy as np
import cv2 as cv
import time


def get_stereo_depth(count):
    remaps = np.load('stereo-rectified-maps.npz')
    left_map_x, left_map_y, right_map_x, right_map_y = remaps['left_map_x'], remaps['left_map_y'], remaps['right_map_x'], remaps['right_map_y']
    frame_pair_dir = 'paired_images'
    images = [('{}/camera0_{}.png'.format(frame_pair_dir, cap_count), '{}/camera1_{}.png'.format(frame_pair_dir, cap_count)) for cap_count in range(count)]
    index = 0

    def nothing(x):
       pass
 
    cv.namedWindow('disp',cv.WINDOW_NORMAL)
    cv.resizeWindow('disp',600,600)
    
    cv.createTrackbar('numDisparities','disp',1,17,nothing)
    cv.createTrackbar('blockSize','disp',5,50,nothing)
    cv.createTrackbar('preFilterType','disp',1,1,nothing)
    cv.createTrackbar('preFilterSize','disp',2,25,nothing)
    cv.createTrackbar('preFilterCap','disp',5,62,nothing)
    cv.createTrackbar('textureThreshold','disp',10,100,nothing)
    cv.createTrackbar('uniquenessRatio','disp',15,100,nothing)
    cv.createTrackbar('speckleRange','disp',0,100,nothing)
    cv.createTrackbar('speckleWindowSize','disp',3,25,nothing)
    cv.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
    cv.createTrackbar('minDisparity','disp',5,25,nothing)
    stereo = cv.StereoBM_create()

    minDisparity, numDisparities = 5, 16
    left_frame = cv.imread(images[index][0])
    right_frame = cv.imread(images[index][1])
    left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
    right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    while True:
        disparity = stereo.compute(cv.cvtColor(left_frame_rect, cv.COLOR_BGR2GRAY), cv.cvtColor(right_frame_rect, cv.COLOR_BGR2GRAY))
        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # display depth map for each image
        # disparity_text = cv.putText(disparity, "Index: {}. Press space to move to the next image. press q to quit".format(index),(100, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, 1)
        cv.imshow("Left Rectified image", left_frame_rect)
        cv.imshow("Disparity", disparity)
        
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
