import os
import pytest
from pathlib import Path
import yaml
import shutil

from calibrator import CalibratorConfig

##flow
##  command line: 

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
        'intrinsic_params_dir': '{}/intrinsic_dir',
        'extrinsic_params_dir': '{}/extrinsic_dir',
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
#from Prof. Libby's stevens google drive:
#research/repoLinks/camera-calibration/camera-calibration.zip
FILE_ID = "1J0AnoTr1EID6_zyAqOS9SI4WkhY0mmbB"


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
    # download calibration images from google drive FILE_ID
    #   into a new folder in cwd called test_images
    download_calibration_images(Path.cwd().joinpath("test_images"), FILE_ID)

    # create a file name: test_calibrator_config.yaml
    #    tmp_path_factory is pytest object, which has a mktemp() function
    #    this next line of code makes a temporary directory, config/
    #      inside the test_data directory, which is the outer temporary directory
    #      being used for this unit test.
    #    then, finally the filename for the yaml file will be inside this temporary dir
    #      camera-calibration/test_data/config/test_calibrator_config.yaml
    yaml_file:Path = tmp_path_factory.mktemp("config") / "test_calibrator_config.yaml"
    
    basedir = yaml_file.parent.parent.relative_to(yaml_file.parent.parent.parent)
    # update test data directory
    config_dict["camera_calibration"]["intrinsic_params_dir"] = config_dict["camera_calibration"]["intrinsic_params_dir"]\
        .format(basedir)
    config_dict["camera_calibration"]["extrinsic_params_dir"] = config_dict["camera_calibration"]["extrinsic_params_dir"]\
        .format(basedir)
    # dump config yaml into a tmp file
    #   dictionary -> yaml data
    yaml_data = yaml.dump(config_dict) 

    ##write yaml data to yaml temp file
    with open(yaml_file, "+w") as f:
        f.write(yaml_data)
        f.close()

    ##the yield will wait for all the tests to finish
    ##  we'll come back to this and look at it a little more closely once we have
    ##  more than one unit test.
    yield yaml_file
    print("Delete tmp files...")
    
    ##delete all temporary files before exiting the unit test
    ##  shutil python package for command line functions
    ##  rmtree is a recursive rm of a directory
    ##  yaml_file = camera-calibration/test_data/config/test_calibrator_config.yaml
    ##  yaml_file.parent.parent = camera-calibration/test_data/
    shutil.rmtree(basedir)
    print("fnished.")
