import cv2 as cv
import os

def capture_single_image(cam_id, count):
    dir = 'single_images/{}'.format(cam_id)

    if not os.path.exists('single_images'):
        os.mkdir('single_images')
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    cap = cv.VideoCapture(cam_id)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    start_capture = False
    cap_count = 0
    cooldown = 100
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
            start_frame = cv.putText(frame, "Press space to begin capturing images; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, 1)
            cv.imshow("Frame", start_frame)

        if start_capture:
            if cooldown == 0:
                cv.imwrite('{}/camera{}_{}.png'.format(dir, cam_id, cap_count), frame)
                cooldown = 50
                cap_count += 1
            text_frame = cv.putText(frame, "Num of images captured: {}\n Countdown: {}".format(cap_count, cooldown),(100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, 1)
            cv.imshow("Frame", text_frame)
            cooldown -= 1
            if cap_count >= count:
                break
    cap.release()

def capture_paired_images(left_cam, right_cam, count):
    dir = 'paired_images'

    if not os.path.exists('paired_images'):
        os.mkdir('paired_images')
    
    left_cap = cv.VideoCapture(left_cam)
    right_cap = cv.VideoCapture(right_cam)

    if not (left_cap.isOpened() and right_cap.isOpened()):
        print("Cannot read video frames")
        exit()

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
            left_start_frame = cv.putText(left_frame, "Press space to begin capturing images; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, 1)
            right_start_frame = cv.putText(right_frame, "Press space to begin capturing images; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2, 1)
            cv.imshow("Left Frame", left_start_frame)
            cv.imshow("Right Frame", right_start_frame)
        
        if start_capture:
            if cooldown == 0:
                cv.imwrite('{}/camera0_{}.png'.format(dir, cap_count), left_frame)
                cv.imwrite('{}/camera1_{}.png'.format(dir, cap_count), right_frame)
                cooldown = 50
                cap_count += 1
            left_text_frame = cv.putText(left_frame, "Num of images captured: {}\n Countdown: {}".format(cap_count, cooldown),(100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, 1)
            right_text_frame = cv.putText(right_frame, "Num of images captured: {}\n Countdown: {}".format(cap_count, cooldown),(100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, 1)
            cv.imshow("Left Frame", left_text_frame)
            cv.imshow("Right Frame", right_text_frame)
            cooldown -= 1
            if cap_count >= count:
                break
    cv.destroyAllWindows()
    left_cap.release()
    right_cap.release()
