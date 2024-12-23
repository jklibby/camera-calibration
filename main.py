import click

from calibrator import StereoCalibrator


@click.command
@click.argument('config-file', type=click.Path(exists=True))
@click.option("--full", default=False, help="Run the whole pipeline.")
@click.option('--capture-single-images', default=False, help="Capture images for single camera calibration.")
@click.option('--capture-stereo-images', default=False, help="Capture images for stereo camera calibration.")
@click.option('--calibrate-single-cameras', default=False, help="Calibrate single cameras.")
@click.option('--calibrate-stereo-cameras', default=False, help="Calibrate stereo cameras.")
@click.option('--rectify-stereo-cameras', default=False, help="Rectify stereo images.")
@click.option('--validate-calibration', default=False, help="Measure the validation checkerboard with stereo calibration. Utilizes DLT.")
@click.option('--tune-disparity', default=False, help="Tune BM and SGBM diaprity params for rectified images.")
@click.option('--drop-stereo-points', default=False, help="Select idnetical points in stereo frames and project them to 3D")
@click.option('--capture-rectified-images', default=False, help="Capture rectified stereo images")
def run(config_file, full, capture_single_images, capture_stereo_images, calibrate_single_cameras, calibrate_stereo_cameras, rectify_stereo_cameras, validate_calibration, tune_disparity, drop_stereo_points, capture_rectified_images):
    ##static function, returnes calibration object, c, with the config file loaded
    calibrator = StereoCalibrator.from_yaml(config_file)

    if capture_single_images or full:
        calibrator.capture_single_camera(calibrator.left_camera_capture)
        calibrator.capture_single_camera(calibrator.right_camera_capture)

    if calibrate_single_cameras or full:
        ##intrinsic calibration for left camera
        ##  this calls cv.calibrateCamera and repojection_error
        calibrator.calibrate_single_camera(calibrator.left_camera_calibrate)
        ##intrinsic calibration for right camera
        calibrator.calibrate_single_camera(calibrator.right_camera_calibrate)
    
    if capture_stereo_images or full:
        calibrator.capture_stereo_camera()

    if calibrate_stereo_cameras or full:
        ##extrinsic calibration
        calibrator.calibrate_stereo_camera()
    
    if rectify_stereo_cameras or full:
        ##rectification
        calibrator.stereo_rectify() 
    
    if validate_calibration or full:
        calibrator.visualize_checkerboards()
        # width, height = calibrator.measure_checkerboard()
    
    if tune_disparity or full:
        calibrator.tune_dispairty()
    
    if capture_rectified_images:
        calibrator.capture_rectified_stereo_images()

    if drop_stereo_points:
        calibrator.point_cloud_selector()

if __name__ == '__main__':
    run()
