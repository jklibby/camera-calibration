import click

from capture_images import capture_single_image, capture_paired_images, get_points_of_interest
from camera_calibration import single_camera_calibrate, single_camera_pose_estimation
from stereo_calibration import stereo_camera_calibrate, stereo_rectification
from stereo_depth import get_stereo_depth, live_stereo_depth, get_object_measurement
from stereo_distance import get_3d_points

PATTERN = (7, 7)

@click.command
@click.option("--capture-single", default=0)
@click.option("--capture-paired", default=0)
@click.option("--calibrate-camera", default=0)
@click.option("--calibrate-camera-path", default='single_images')
@click.option("--detect-pose", default=False)
@click.option("--calibrate-stereo-camera", default=0)
@click.option("--calibrate-stereo-camera-path", default='paired_images')
@click.option("--calibrate-stereo-depth", default=0)
@click.option("--stereo-depth", default=False)
@click.option("--distance", default=False)
def run(capture_single, capture_paired, calibrate_camera, calibrate_camera_path, detect_pose, calibrate_stereo_camera, calibrate_stereo_camera_path, calibrate_stereo_depth, stereo_depth, distance):
    # cam_id is hard coded, maybe use a config file?
    cam_left = 0
    cam_right = 1
    if capture_single != 0:
        capture_single_image(cam_left, count=capture_single)
        capture_single_image(cam_right, count=capture_single)
        
    if capture_paired != 0:
        capture_paired_images(cam_left, cam_right, count=capture_paired)

    if (calibrate_camera != 0 or capture_single != 0):
        single_camera_calibrate(cam_left, count=calibrate_camera, path=calibrate_camera_path)
        single_camera_calibrate(cam_right, count=calibrate_camera, path=calibrate_camera_path)
        
    if detect_pose:
        single_camera_pose_estimation(cam_left, (3, 4))
        single_camera_pose_estimation(cam_right, (3, 4))
    
    if calibrate_stereo_camera !=0 :
        stereo_camera_calibrate(calibrate_stereo_camera, calibrate_stereo_camera_path, PATTERN)
        stereo_rectification(calibrate_stereo_camera, calibrate_stereo_camera_path, PATTERN)
    
    if calibrate_stereo_depth != 0:
        get_stereo_depth(calibrate_stereo_depth, path=calibrate_stereo_camera_path)
    if stereo_depth:
        get_object_measurement(cam_left, cam_right, 50)
    
    if distance:
        p1, p2 = get_points_of_interest()
        get_3d_points(p1, p2)

if __name__ == '__main__':
    run()