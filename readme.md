# Stereo Camera - Depth Map

This repository contains code to calibrate and create depths for a stereo camera system. To begin calibrating run the following commands. 

The calibration will depend on a (7x7) chessboard pattern. 

**Python Version - 3.10**

```
Usage: main.py [OPTIONS] CONFIG_FILE

Options:
  --full BOOLEAN                  Run the whole pipeline.
  --capture-single-images BOOLEAN Capture images for single camera calibration.
  --capture-stereo-images BOOLEAN Capture images for stereo camera calibration.
  --calibrate-single-cameras BOOLEAN Calibrate single cameras.
  --calibrate-stereo-cameras BOOLEAN Calibrate stereo cameras.
  --rectify-stereo-cameras BOOLEAN Rectify stereo images.
  --validate-calibration BOOLEAN  Measure the validation checkerboard with stereo calibration. Utilizes DLT.
  --tune-disparity BOOLEAN        Tune BM and SGBM diaprity params for rectified images.
  --help                          Show this message and exit.   
```

The command below, will capture 10 single camera images for both the left camera and the right camera. It will then capture 10 paired images for both left camera and right camera. It will then use 10 images to calibration single camera, left and right. Then it will use 10 paired images to calibrate stereo cameras. After that it will display recitified images for the cameras. Then it will display Depth Maps with a wondow to calibrate hyperparameters. 

```
python main.py --capture-single=10 --capture-paired=10 --calibrate-camera=10 --calibrate-stereo-camera=10 --stereo-depth=10
```



## Capturing Images

To calibrate a stereo system and generate depth maps, images must be captured from each camera separately and together at the same moment.

### Single Camera Image Capture

Single camera images will be stored in the `single_images` folder, within subdirectories `0` and `1`, referring to the left and right cameras respectively.

Once the window displaying the live feed from the camera pops up, press the space button and wait for the countdown to begin. At 0, the frame will be automatically stored. This allows ample time to orient the frame in different ways.

![Single Camera Image Capture](screenshots/Capture-Single-Frame.png)

### Stereo Camera Image Capture

Stereo camera images will be stored in the `paired_images` folder. The nomenclature is `camera{camera_id}_{capture_frame_count}.png`. Each capture will produce a pair of images.

Once the window displaying the live feed from the camera pops up, press the space button and wait for the countdown to begin. At 0, the frame will be automatically stored. Ensure the chessboard is in frame for both cameras.

![Stereo Camera Image Capture](screenshots/Capture-Stereo-Frame.png)

## Calibration

After capturing the frames, proceed to calibrate the single camera and stereo camera systems.

### Single Camera Calibration

A new window will display the chessboard corners highlighted in a rainbow color scheme. Select the grids with similar orientation to minimize the RMSE. The RMSE should lie between 0 and 1, with a good value being less than 0.5.

![Single Camera Image Calibration](screenshots/Calibrate-Single-Camera.png)

### Stereo Camera Calibration

A new window will display the chessboard corners highlighted in a rainbow color scheme for both left and right frames. Select the grids with similar orientation to minimize the RMSE. The RMSE should lie between 0 and 1, with a good value being less than 0.5.

Press 's' to skip the current frames, press space to include the current frame. Skip a bad frame to improve RMSE.

Good stereo match 
![Good Stereo Camera Image Calibration](screenshots/Good-Stereo-Match.png)
bad stereo match
![Bad Stereo Camera Image Calibration](screenshots/Bad-Stereo-Match.png)

## Stereo Rectification

Once the system is calibrated to a low enough RMSE, rectification maps will be created to remove distortions from the images.

## Stereo Depth Map

After calculating the rectification maps, depth maps can be computed. The depth maps need to be calibrated using various hyperparameters, which can be adjusted using trackbars in a new window. Press `c` to apply changes, `space` to move to the next image, and `q` to quit.
