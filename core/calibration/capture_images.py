import cv2 as cv
import numpy as np
import os
from pathlib import Path

from options import SingleCameraCaptureOptions, StereoCameraCaptureOptions

def capture_single_image(opts: SingleCameraCaptureOptions):
    """
        Function to capture single camera images and store them in `opts.image_path`

        Args:
            opts (SingleCameraCaptureOptions): Options for capturing single camera options. 
    """
    cam_id = opts.cam_id
    count = opts.count
    dir = Path(opts.path).absolute()

    if not Path.exists(dir):
        Path.mkdir(dir, parents=True)


    cap = cv.VideoCapture(cam_id)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, opts.cv_options.resolution[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, opts.cv_options.resolution[1])

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    start_capture = False
    cap_count = 0
    cooldown = 100

    opts.cv_options.named_window("Frame")
    while True:
        ret, frame = cap.read()
        key = cv.waitKey(1)
        
        if not ret:
            print("Cannot read video frames")
            cap.released()
            break
        
        if key & 0xFF == ord(' '):
            start_capture = True
        
        if key & 0xFF == ord('q'):
            break
        
        if not start_capture:
            start_frame = cv.putText(frame, "Press space to begin capturing images; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
            cv.imshow("Frame", start_frame)

        if start_capture:
            if cooldown == 0:
                img_path = str(dir.joinpath('camera{}_{}.png'.format(cam_id, cap_count)))
                cv.imwrite(img_path, frame)
                cooldown = 50
                cap_count += 1
            text_frame = cv.putText(frame, "Num of images captured: {}\n Countdown: {}".format(cap_count, cooldown),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, 1)
            cv.imshow("Frame", text_frame)
            cooldown -= 1
            if cap_count >= count:
                break
    cap.release()
    cv.destroyAllWindows()

def capture_paired_images(opts: StereoCameraCaptureOptions):
    """
        Function to capture single camera images and store them in `opts.path`

        Args:
            opts (StereoCameraCaptureOptions): Options for capturing single camera options. 
    """
    dir = Path(opts.path)
    left_cam = opts.left_cam_id
    right_cam = opts.right_cam_id
    count = opts.count

    if not Path.exists(dir):
        Path.mkdir(dir, parents=True)
    
    left_cap = cv.VideoCapture(left_cam)
    right_cap = cv.VideoCapture(right_cam)

    left_cap.set(cv.CAP_PROP_FRAME_WIDTH, opts.cv_options.resolution[0])
    left_cap.set(cv.CAP_PROP_FRAME_HEIGHT, opts.cv_options.resolution[1])

    right_cap.set(cv.CAP_PROP_FRAME_WIDTH, opts.cv_options.resolution[0])
    right_cap.set(cv.CAP_PROP_FRAME_HEIGHT, opts.cv_options.resolution[1])

    if not (left_cap.isOpened() and right_cap.isOpened()):
        print("Cannot read video frames")
        exit()

    opts.cv_options.named_window("Left Frame")
    opts.cv_options.named_window("Right Frame")

    start_capture = False
    cooldown = 100
    cap_count = 0
    while True:
        lret, left_frame = left_cap.read()
        rret, right_frame = right_cap.read()
        key = cv.waitKey(1)
        
        if not (rret and lret):
            print("Cannot read video frames")
            left_cap.released()
            right_cap.release()
            break
        
        if key & 0XFF == ord(' '):
            start_capture = True
        
        if key & 0XFF == ord('q'):
            break

        if not start_capture:
            left_start_frame = cv.putText(left_frame, "Press space to begin capturing images; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
            right_start_frame = cv.putText(right_frame, "Press space to begin capturing images; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, 1)
            cv.imshow("Left Frame", left_start_frame)
            cv.imshow("Right Frame", right_start_frame)
        
        if start_capture:
            if cooldown == 0:
                cv.imwrite('{}/camera0_{}.png'.format(dir, cap_count), left_frame)
                cv.imwrite('{}/camera1_{}.png'.format(dir, cap_count), right_frame)
                cooldown = 50
                cap_count += 1
            left_text_frame = cv.putText(left_frame, "Num of images captured: {}\n Countdown: {}".format(cap_count, cooldown),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, 1)
            right_text_frame = cv.putText(right_frame, "Num of images captured: {}\n Countdown: {}".format(cap_count, cooldown),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, 1)
            cv.imshow("Left Frame", left_text_frame)
            cv.imshow("Right Frame", right_text_frame)
            cooldown -= 1
            if cap_count >= count:
                break
    cv.destroyAllWindows()
    left_cap.release()
    right_cap.release()

