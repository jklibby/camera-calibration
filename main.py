import click

from capture_images import capture_single_image, capture_paired_images
from camera_calibration import single_camera_calibrate
from stereo_calibration import stereo_camera_calibrate, stereo_rectification
from stereo_depth import get_stereo_depth

@click.command
@click.option("--capture-single", default=0)
@click.option("--capture-paired", default=0)
@click.option("--calibrate-camera", default=0)
@click.option("--calibrate-stereo-camera", default=0)
@click.option("--stereo-depth", default=0)
def run(capture_single, capture_paired, calibrate_camera, calibrate_stereo_camera, stereo_depth):
    # cam_id is hard coded, maybe use a config file?
    cam_left = 0
    cam_right = 1
    if capture_single != 0:
        capture_single_image(cam_left, count=capture_single)
        capture_single_image(cam_right, count=capture_single)

    if capture_paired != 0:
        capture_paired_images(cam_left, cam_right, count=capture_paired)

    if (calibrate_camera != 0 or capture_single != 0):
        single_camera_calibrate(cam_left, count=calibrate_camera)
        single_camera_calibrate(cam_right, count=calibrate_camera)
    
    if calibrate_stereo_camera !=0 :
        stereo_camera_calibrate(calibrate_stereo_camera)
        stereo_rectification(calibrate_stereo_camera)
    
    if stereo_depth != 0:
        get_stereo_depth(stereo_depth)

if __name__ == '__main__':
    run()