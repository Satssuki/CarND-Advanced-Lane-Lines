import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


# Define a class to receive the characteristics of each line detection
class LaneLine(object):
    def __init__(self, fit_coeff=None):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = fit_coeff
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def walk_lines(binary_warped, prevLeftXBase = None, prevRightXBase=None):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, out_img

def bin_thresh_img(img, thresh):
    binary_img = np.zeros_like(img)
    binary_img[(img >= thresh)] = 1
    return binary_img

def plot_lane_walk(binary_warped, left_fit, right_fit, out_img):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    h, w, ch = out_img.shape

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, w)
    plt.ylim(h, 0)


def discover_two_lines(warped):
    '''
    takes an image that has been thresholded to include mostly lane pixels
    and transformed to remove most of persepective warping
    do a fresh scan of an image to determine where two lanes lines might be
    assumes that we are already in the center of the lines
    '''
    left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = walk_lines(warped)

    left_lane = LaneLine(left_fit)
    right_lane = LaneLine(right_fit)

    return left_lane, right_lane


def get_pixel_to_m(left_lane_x, right_lane_x, out_img_shape):
    '''
    get the scale of pixel in world space (meters)
    '''
    h, w = out_img_shape
    
    # Define conversions in x and y from pixels space to meters
    pixel_lane_width = right_lane_x - left_lane_x
    meters_lane_width = 3.7    
    
    # meters per pixel in x dimension
    xm_per_pix = meters_lane_width / pixel_lane_width
    
    # meters per pixel in y dimension
    ym_per_pix = 20.0 /  h
    
    return xm_per_pix, ym_per_pix


def sample_curve(coeff, val):
    '''
    given a polynomial order 2, sample curve at val
    and return value on the line
    '''
    return coeff[0]*val**2 + coeff[1]*val + coeff[2]


def sample_curves_to_points(left_fit, right_fit, warped_shape):
    '''
    sample the points of two y paramatized curves in an image
    returns:
        ploty, the array of y values
        leftX, the array of x values sampled from left_fit curve
        rightX, the array of x values sampled from right_fit curve    
    '''
    h, w = warped_shape
    
    #sample curve
    ploty = np.linspace(0, h-1, num=h)
    leftX = []
    rightX = []
    
    for val in range(0, h):
        lX = sample_curve(left_fit, val)
        rX = sample_curve(right_fit, val)
        leftX.append(lX)
        rightX.append(rX)
        
    leftX = np.array(leftX)
    rightX = np.array(rightX)
    return ploty, leftX, rightX


def get_world_space_rad_curve_m(left_fit, right_fit, warped_shape):
    '''
    get the radius of curvature in world space (meters)
    '''
    h, w = warped_shape
    
    #sample curve
    ploty, leftX, rightX = sample_curves_to_points(left_fit, right_fit, warped_shape)

    xm_per_pix, ym_per_pix = get_pixel_to_m(leftX[-1], rightX[-1], warped_shape)

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = h
    
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftX*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightX*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


def unwarp_lines(warped, left_fit, right_fit, Minv, orig_image_shape):
    
    #sample the curves for points
    ploty, left_fitx, right_fitx = sample_curves_to_points(left_fit, right_fit, warped.shape)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    h, w = warped.shape
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_image_shape[1], orig_image_shape[0])) 
    
    return newwarp

def get_lane_deviation(left_fit, right_fit, warped_shape):
    '''
    get the deviation from center of lane in world space (meters)
    '''
    h, w = warped_shape
    
    #sample curve
    leftX = sample_curve(left_fit, h)
    rightX = sample_curve(right_fit, h)
    centerX = w / 2.

    xm_per_pix, ym_per_pix = get_pixel_to_m(leftX, rightX, warped_shape)

    leftM = leftX * xm_per_pix
    rightM = rightX * xm_per_pix
    centerM = centerX * xm_per_pix
    centerLaneM = (rightM - leftM) / 2.0 + leftM
    return centerM - centerLaneM


def get_lane_info(left_lane, right_lane, warped_shape):
        
    l_rad, r_rad = get_world_space_rad_curve_m(
                            left_lane.current_fit, 
                            right_lane.current_fit, 
                            warped_shape)

    curvature = (l_rad + r_rad) / 2.

    deviation = get_lane_deviation(
                            left_lane.current_fit, 
                            right_lane.current_fit, 
                            warped_shape)

    return curvature, deviation


def write_lane_info(left_lane, right_lane, img, warped_shape):
    '''
    write lane pos and curve stats onto image frame
    '''
        
    curvature, deviation = get_lane_info(left_lane, right_lane, warped_shape)

    curvature_text = 'Lane Curvature: {:.2f} m'.format(curvature)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, curvature_text, (50, 50), font, 1, (221, 28, 119), 2)

    deviation_info = 'Lane Deviation: {:.3f} m'.format(deviation)
    cv2.putText(img, deviation_info, (50, 90), font, 1, (221, 28, 119), 2)


def get_lane_overlay(warped, left_lane, right_lane, invM, orig_image_shape):
    '''
    takes an image that has been thresholded to include mostly lane pixels
    and transformed to remove most of persepective warping
    takes two lane lines, and the invM tranform back to original image space
    and return an image of the lane overlay
    '''

    newwarp = unwarp_lines(warped, 
        left_lane.current_fit, 
        right_lane.current_fit, 
        invM,
        orig_image_shape)
    
    return newwarp

def update_two_lines(warped, left_lane, right_lane):
    '''
    a more efficient update of lanes using last known good fit
    to speed search
    '''
    left_lane.current_fit, right_lane.current_fit = update_walk_lines(
        warped, 
        left_lane.current_fit, 
        right_lane.current_fit)

    return left_lane, right_lane
    
# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
def update_walk_lines(binary_warped, left_fit, right_fit, debug=False):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if debug:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit, right_fit

