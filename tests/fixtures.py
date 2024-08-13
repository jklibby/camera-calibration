import pytest
from pathlib import Path
import yaml

from calibrator import CalibratorConfig

##check if images exist. if not, images are downloaded.
from .download_camera_calibration import download_calibration_images

config_dict: CalibratorConfig = {
    'cv_options': {
        'resolution': [1920, 1080],
        'window_size': [1280, 720]
    },
    'camera_capture': {
        'left_cam_id': 0,
        'right_cam_id': 1, 
        'image_count': 50, 
        'image_path': 'test_images/calibration-images/calibration-images/single_images/',
        'stereo_images_path': 'test_images/calibration-images/calibration-images/stereo_images/', 
        'stereo_images_count': 50
    },
    'camera_calibration': {
        'intrinsic_params_dir': 'test_data/intrinsic_dir',
        'extrinsic_params_dir': 'test_data/extrinsic_dir',
        'pattern_size': [10, 7],
        'world_scaling': 1.5,
        'single_calibration_error_threshold': 0.3,
        'stereo_calibration_error_threshold': 0.7, 
        'headless': True
    },
    'validation': {
        'pattern_size': [10, 7]
    }
}

#This is the file id for the google drive location of the checkerboard images.
FILE_ID = "11DnV4eJ260KHSar1_fIcYjrdB4gLQn7g"

##when pytest gets called, it will first call fixturs.py
#   and then calibration_yaml_fixture() will be called
#   then, later, when test_* gets called and calibration_yaml_fixture
#     is passed in as it's parameter, it will return the yaml_file
#     which is yielded here at the end of this function.
#   The whole point of this is so that we can have this yaml file
#     which only exists temporarily in this session.
#   The values of that yaml file are as defined above in config_dict
#   Another way to implement this would be to create the dictionary 
#     at the top of test_calibrator.py, before any test_* functions,
#     but it wouldn't be as clean because it's not clear what the scope is.
@pytest.fixture(scope="session")
def calibration_yaml_fixture(tmp_path_factory):
    # download calibration images
    download_calibration_images(Path.cwd().joinpath("test_images"), FILE_ID)

    # dump config yaml into a tmp file
    yaml_data = yaml.dump(config_dict)
    yaml_file:Path = tmp_path_factory.mktemp("config") / "test_calibrator_config.yaml"
    with open(yaml_file, "+w") as f:
        f.write(yaml_data)
        f.close()
    yield yaml_file
    
