
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
<img src="./data/test_cal/test_image.jpg" alt="distorted" width="300"> <img src="./data/test_cal/test_undist.jpg" width="300">

## 2. Apply a distortion correction to raw images.

![street image](./data/test_cal/straight_lines1.jpg =300x)
![street image undistorted](./data/test_cal/straight_lines1_undist.jpg =300x)

## 3. Create a thresholded binary image

![street image binary](./data/test_cal/straight_lines1_binary.jpg =300x)


## 4. Apply a perspective transformation

![bird eye view](./data/test_cal/straight_lines1_bird_eye.jpg =300x)

## 5. Fit a polynomial 

![polynomial](./data/test_cal/straight_lines1_poli.jpg =600x)


## 6. Mark the detected lane in the orignal image

![image with lane](./data/test/../test_cal/straight_lines1_lane.jpg =300x)

