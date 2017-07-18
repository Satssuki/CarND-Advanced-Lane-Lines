## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Final Result Video

[![IMAGE ALT TEXT](https://img.youtube.com/vi/_tIMUhJ2D54/0.jpg)](https://www.youtube.com/watch?v=_tIMUhJ2D54 "Advanced Lane Finding Video.")

Video file is available [here](https://github.com/tawnkramer/CarND-Advanced-Lane-Lines/blob/master/project_video_out.mp4).

[//]: # (Image References)

[image1]: ./examples/undistort.png "Undistorted"
[image2]: ./examples/hls.png "hls"
[image3]: ./examples/rgb.png "rgb"
[image4]: ./examples/yuv.png "yuv"
[image5]: ./examples/custom_kernel_source.png "kernel_source"
[image6]: ./examples/custom_kernel.png "kernel_img"
[image7]: ./examples/combined_masks.png "combined"
[image8]: ./examples/thresholding_results.png "thresh_res"
[image9]: ./examples/sobel_gradients.png "sobel"
[image10]: ./examples/persp_transform.png "persp"
[image11]: ./examples/histogram_walk.png "histogram"
[image12]: ./examples/polynomial.png "polynomial"
[image13]: ./examples/result_1.png "result_1"
[image14]: ./examples/result_2.png "result_2"
[image15]: ./examples/result_3.png "result_3"

### Source Code
All source code can be viewed in the python notebook file [AdvancedLaneFinding.ipynb](https://github.com/tawnkramer/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneFinding.ipynb) or in python source [here](https://github.com/tawnkramer/CarND-Advanced-Lane-Lines/tree/master/src).

### Camera Calibration

#### 1. Computed the camera matrix and distortion coefficients. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline

#### 1. Correct Image Distortion

Example above.

#### 2. Pixel value thresholding in different color spaces

The source image was converted to HLS, RGB, and YUV color spaces. Various threshold values were hand-tuned to select just a portion of the image that would contribute to finding lane pixels.

##### HLS Color space example
![alt text][image2]
##### RGB Color space example
![alt text][image3]
##### YUV Color space example
![alt text][image4]

#### 3. Custom Convolutional Kernel

I developed a custom convolutional kernel to attempt to directly detect diagonal lines. This is derived from the standard Sobel filter for X and Y edge gradients. 
![alt text][image5]

![alt text][image6]

#### 4. Mask Region of Intrest

A custom polygon was created to exclude pixels outside the road.
![alt text][image7]

#### 5. Combined Masks

Individual masks were thresholded to a floating point image where each pixel was either 0 or 0.1. Then the masks were added together and all pixels with two or more contributions were used.

![alt text][image8]

An experiment to Sobel Gradients did not result in useful information, and was ultimately discarded. Here's an example of that filter.
![alt text][image9]

#### 6. Perspective Transform

A perspective transform was used to rectify the binary image. The attempted to transform pixels in the source image to remove the effects of camera perspective such that parallel lines in the world result in parallel image lines.

To accomplish this, a custom polygon was chosen which matched the lane lines in the perspective view. Then a second polgon was chosen as the destination space. This was more rectangular, but not perfectly so. The OpenCV function getPerspectiveTransform() was used to calculate the matrix. And OpenCV warpPerspective() was used to apply the matrix to the image.

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]


