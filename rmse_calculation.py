import os
import numpy as np
import cv2 as cv

from calibration import capture_single_image, capture_paired_images
from calibration import single_camera_calibrate
from calibration import stereo_camera_calibrate, stereo_rectification
from depth_estimation import get_object_measurement, get_checkerboard_3d

def start(count, index, single_flags, paired_flags, capture_single=False, capture_paired=False):
    single_path = "single_images"
    paired_path = "paired_images"
    if capture_single[0]:
        capture_single_image(0, count)
    if capture_single[1]:
        capture_single_image(1, count)
    s_filename = "f_{}_{}_{}_{}".format(count, single_flags or 0, paired_flags or 0, index)
    rmse_left = single_camera_calibrate(0, count, path=single_path, pattern_size=(10, 7), flags=single_flags, dir=s_filename, refine_val=0.25)
    rmse_right = single_camera_calibrate(1, count, path=single_path, pattern_size=(10, 7), flags=single_flags, dir=s_filename, refine_val=0.25)
    if capture_paired:
        capture_paired_images(0, 1, count)
    rmse = stereo_camera_calibrate(count, paired_path, (10, 7), flags=paired_flags, dir=s_filename, skip_gui=True, refine_val=0.5)
    stereo_rectification(count, paired_path, dir=s_filename)

    # corners_3d = get_checkerboard_3d(count, (10, 7), s_filename)
    # np.save("3d_corners", corners_3d)
    # project points and calculate distance
    measured_error = get_object_measurement(0, 1, (3, 4), dir=s_filename)
    return count, rmse, rmse_left, rmse_right, measured_error

total_count = 100
step = 10

single_flags = (cv.CALIB_RATIONAL_MODEL)
paired_flags = (cv.CALIB_FIX_INTRINSIC + cv.CALIB_ZERO_DISPARITY + cv.CALIB_RATIONAL_MODEL)

index = 0
results_file = "results/results_{}_{}_{}.npz".format(single_flags, paired_flags, index)
results_file_name = "results/results_{}_{}_{}".format(single_flags, paired_flags, index)
while True:
    if not os.path.exists(results_file):
        break
    index += 1
    results_file = "results/results_{}_{}_{}.npz".format(single_flags, paired_flags, index)
    results_file_name = "results/results_{}_{}_{}".format(single_flags, paired_flags, index)
        
res_1 = start(total_count, index, single_flags, paired_flags, capture_single=(False, False), capture_paired=False)

results = [res_1]
print(res_1)
print("*"* 150)
# for i in range(30, total_count, step):
#     res = start(i, index, single_flags, paired_flags, capture_single=(False, False), capture_paired=False)
#     results.append(res)
#     print(res)
#     print("*"* 150)

np.savez(results_file_name, results=results)
