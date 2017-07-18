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

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]


