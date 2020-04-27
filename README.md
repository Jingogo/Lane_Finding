
# Lane Finding Project

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.


## 1. Camera Calibration
<img src="./data/test_cal/test_image.jpg" width="300"> <img src="./data/test_cal/test_undist.jpg" width="300">

## 2. Apply a distortion correction to raw images.
<img src="./data/test_cal/straight_lines1.jpg" width="300"><img src="./data/test_cal/straight_lines1_undist.jpg" width="300">

## 3. Create a thresholded binary image
<img src="./data/test_cal/straight_lines1_binary.jpg" width="300">

## 4. Apply a perspective transformation
<img src="./data/test_cal/straight_lines1_bird_eye.jpg" width="300">

## 5. Fit a polynomial 
<img src="./data/test_cal/straight_lines1_poli.jpg" width="300">

## 6. Mark the detected lane in the orignal image
<img src="./data/test/../test_cal/straight_lines1_lane" width="300">

