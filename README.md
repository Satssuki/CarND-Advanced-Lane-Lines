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
[image11a]: ./examples/histogram.png "histogram"
[image11]: ./examples/histogram_walk.png "histogram_walk"
[image12]: ./examples/polynomial.png "polynomial"
[image13]: ./examples/result_1.png "result_1"
[image14]: ./examples/result_2.png "result_2"
[image15]: ./examples/result_3.png "result_3"
[image16]: ./output_images/test3.jpg "lane_info"

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
The first row shows the result of the color space conversion. The second row shows the resulting mask from selecting pixels at a certain threshold.
![alt text][image2]
##### RGB Color space example
The first row shows the result of the color space conversion. The second row shows the resulting mask from selecting pixels at a certain threshold.
![alt text][image3]
##### YUV Color space example
The first row shows the result of the color space conversion. The second row shows the resulting mask from selecting pixels at a certain threshold.
![alt text][image4]

#### 3. Custom Convolutional Kernel

I developed a custom convolutional kernel to attempt to directly detect right sloping diagonal lines. This is derived from the standard Sobel filter for X and Y edge gradients. The kernel was flipped horizontally to detect left sloping edges.
![alt text][image5]

Here are examples of resulting masks:
![alt text][image6]

An experiment to use Sobel gradients did not result in useful information, and was ultimately discarded. Here's an example of that filter.
![alt text][image9]

#### 4. Combined Masks

Individual masks were thresholded to a floating point image where each pixel was either 0 or 0.1. Then the masks were added together and all pixels with two or more contributions were used. Here the full image is shown without the region of interest applied.

![alt text][image7]

#### 5. Mask Region of Intrest

A custom polygon was created to exclude pixels outside the road.
![alt text][image8]


#### 6. Perspective Transform

A perspective transform was used to rectify the binary image. This attempts to transform pixels in the source image to remove the effects of camera perspective such that parallel lines in the world result in parallel image lines.

To accomplish this, a custom polygon was chosen which matched the lane lines in the perspective view. Then a second polgon was chosen as the destination space. This was more rectangular, but not perfectly so. The OpenCV function `cv2.getPerspectiveTransform()` was used to calculate the matrix. And `cv2.warpPerspective()` was used to apply the matrix to the image.

![alt text][image10]

#### 7. Identify Lane Pixels with Histogram

I created a histogram of pixels at each column of the image. Then the two largest regions to the right and left of centerline were taken as the starting lane positions.

![alt text][image11a]

This histogram was moved in progressive bounding boxes up the image. The bounding box was adjusted to the center of the detected lane position. This gives it continuity. Here's the example of the binary mask and resulting walk of the image showing the bounding boxes at each step.

![alt text][image11]

#### 8. Fit Polynomial

A second order polynomial in the Y axis was fit to the resulting lane points using `numpy.polyfit`. Here's an image displaying the resulting curve overlayed over the binary mask.

![alt text][image12]

#### 9. Inverse Perspective Transform

The resulting curves were transformed back into the original image space using the inverse of the matrix which went to linear space. Then the curved green polygon was overlayed over the original lane image. Here are three results:

![alt text][image13]
![alt text][image14]
![alt text][image15]

#### 10. Lane Curvature and Deviation

The curves were transform from pixel space to world space using assumed lane width of 3.7 m and view distance polygon of 20 m. This curve was sampled at the bottom edge closest to the car at both lane lines. The curvature was taken as the derivative of the curve function at that point, and the values averaged to determine a radius.

The deviation, or lane position, was calculated by using the same pixel to world space transform. It assumed the camera was mounted at the center of the car, and thus determined the center X offset from the lane positions.

Here's an example showing lane info and curvature overlay:

![alt text][image16]

