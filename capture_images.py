import cv2 as cv
import numpy as np
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
            start_frame = cv.putText(frame, "Press space to begin capturing images; q to quit",(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
            cv.imshow("Frame", start_frame)

        if start_capture:
            if cooldown == 0:
                cv.imwrite('{}/camera{}_{}.png'.format(dir, cam_id, cap_count), frame)
                cooldown = 50
                cap_count += 1
            text_frame = cv.putText(frame, "Num of images captured: {}\n Countdown: {}".format(cap_count, cooldown),(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, 1)
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
            cooldown -= 2
            if cap_count >= count:
                break
    cv.destroyAllWindows()
    left_cap.release()
    right_cap.release()

def get_points_of_interest(left_cam=0, right_cam=1):

    left_cap = cv.VideoCapture(left_cam)
    right_cap = cv.VideoCapture(right_cam)

    if not (left_cap.isOpened() and right_cap.isOpened()):
        print("Cannot read video frames")
        exit()
    
    left_poi = POICollection("left")
    right_poi = POICollection("right")

    while True:
        lret, left_frame = left_cap.read()
        rret, right_frame = right_cap.read()
        key = cv.waitKey(1) & 0xFF
        
        cv.imshow("left", left_frame)
        cv.imshow("right", right_frame)

        if not (rret and lret):
            print("Cannot read video frames")
            left_cap.release()
            right_cap.release()
            break
        
        if key == ord('q'):
            left_cap.release()
            right_cap.release()
            cv.destroyAllWindows()
            break
    
    remaps = np.load('stereo-rectified-maps.npz')
    left_map_x, left_map_y, right_map_x, right_map_y = remaps['left_map_x'], remaps['left_map_y'], remaps['right_map_x'], remaps['right_map_y']
    
    left_frame = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
    right_frame = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
    
    cv.namedWindow("left_image")
    cv.setMouseCallback("left_image", left_poi.add)

    while True:
        cv.imshow("left_image", left_frame)
        if len(left_poi.collection) > 0:
            if left_poi.collection[-1]:
                left_frame = cv.putText(left_frame, '{}'.format(len(left_poi.collection)), left_poi.collection[-1], cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3, 1)
        # cv.imshow("right_image", right_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('w'):
            cv.destroyAllWindows()
            break
    
    cv.namedWindow("right_image")
    cv.setMouseCallback("right_image", right_poi.add)
    colors = [(255, 0, 0), (0, 255, 0)]
    for i, color in enumerate(colors):
        if i >= len(left_poi.collection):
            break
        right_frame = cv.line(right_frame, (0, left_poi.collection[i][1]), (right_frame.shape[1], left_poi.collection[i][1]), color)
    while True:
        cv.imshow("right_image", right_frame)
        if len(right_poi.collection) > 0:
            if right_poi.collection[-1]:
                right_frame = cv.putText(right_frame, '{}'.format(len(right_poi.collection)), right_poi.collection[-1], cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3, 1)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cv.destroyAllWindows()
            break

    cv.destroyAllWindows()
    return(left_poi.to_numpy(), right_poi.to_numpy())

class POICollection(object):
    def __init__(self, window):
        self.window = window
        self.collection = list()
    
    def add(self, event, x, y, flags, param):
        print("event: {}: {}".format(self.window, event), cv.EVENT_LBUTTONDBLCLK)
        if event == cv.EVENT_LBUTTONDOWN:
            self.collection.append((x, y))
    
    def to_numpy(self):
        return np.array(self.collection)