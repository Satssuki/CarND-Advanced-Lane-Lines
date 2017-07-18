'''
functions to do image color processing to find threshold to
extract lane lines
'''
import math
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def thresh_mask(img, thresh, val=0.1):
    flt_img = np.zeros_like(img, dtype=float)
    flt_img[(img > thresh[0]) & (img <= thresh[1])] = val
    return flt_img


def combine_masks(S, U, V, R, G, B, DR, DL, thresh=0.1):    
    '''
    combine some masks
    '''
    combined = np.zeros_like(S, dtype=np.uint8)
    combined[(S + U + R + G + B + DR + DL) > thresh] = 1
    return combined


def diag_masks(img):
    '''
    apply custom conv kernels to detect left and right diagonal lines
    '''
    # construct the Sobel x-axis kernel - diagonal right
    sobelDR = np.array((
        [-1, -1, 0, 0, 1],
        [-2, -1, 0, 0, 2],
        [-2, -1, 0, 0, 2],
        [-1, 0, 0, 0, 2],
        [-1, 0, 0, 1, 1]), dtype="int")
    
    sobelDL = np.fliplr(sobelDR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diagR = cv2.filter2D(gray, -1, sobelDR)
    diagL = cv2.filter2D(gray, -1, sobelDL)
    
    thresh = (50, 255)
    diagRth = thresh_mask(diagR, thresh, val=0.05)
    
    thresh = (50, 255)
    diagLth = thresh_mask(diagL, thresh, val=0.05)
    
    return diagRth, diagLth


def color_thresold_challenge(img):
    '''
    --tuned to values of the challenge video.. still not working well.
    take an BGR image
    splits the image into RGB, HLS, and YUV color spaces
    then applies custom thresholding to get certain parts of image
    then combines the masks into a single mask which we feel is most
    likely to contain lane lines
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(hls)
    
    thresh = (80, 255)
    S = thresh_mask(S, thresh)

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(rgb)

    thresh = (165, 255)
    R = thresh_mask(R, thresh)

    thresh = (165, 255)
    G = thresh_mask(G, thresh)

    thresh = (165, 255)
    B = thresh_mask(B, thresh)
    
    yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    thresh = (50, 100)
    U = thresh_mask(U, thresh)
    
    thresh = (150, 255)
    V = thresh_mask(V, thresh)
    
    DR, DL = diag_masks(img)
    
    return combine_masks(S, U, V, R, G, B, DL, DR)


def color_thresold(img):
    '''
    take an BGR image
    splits the image into RGB, HLS, and YUV color spaces
    then applies custom thresholding to get certain parts of image
    then combines the masks into a single mask which we feel is most
    likely to contain lane lines
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(hls)
    
    thresh = (150, 255)
    S = thresh_mask(S, thresh)

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(rgb)

    thresh = (200, 255)
    R = thresh_mask(R, thresh)

    thresh = (200, 255)
    G = thresh_mask(G, thresh)

    thresh = (220, 255)
    B = thresh_mask(B, thresh)
    
    yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    thresh = (50, 100)
    U = thresh_mask(U, thresh)
    
    thresh = (150, 255)
    V = thresh_mask(V, thresh)
    
    DR, DL = diag_masks(img)
    
    return combine_masks(S, U, V, R, G, B, DL, DR)


def make_mask(img_size, #width, height tuple
              horizon_perc, # the upper threshold, as a percent of height
              bottom_perc, #the lower thresh, as a percent of height
              mask_bottom_perc = 1.0, #the lower percent of width
              mask_top_perc = 0.5): #the upper percent of width
    '''
    make polygon to mask mage
    '''
    img_width = img_size[0]
    img_height = img_size[1]
    
    centerX = img_width / 2
    
    horizon_y = math.floor(horizon_perc * img_height)
    bottom_y_margin = math.floor(bottom_perc * img_height)
    bottom = img_height - bottom_y_margin
    top = horizon_y
    
    mask_bottom_left_x   = math.floor(centerX - img_width * (mask_bottom_perc * 0.5))
    mask_bottom_right_x  = math.floor(centerX + img_width * (mask_bottom_perc * 0.5))
    mask_top_left_x      = math.floor(centerX - img_width * (mask_top_perc * 0.5))
    mask_top_right_x     = math.floor(centerX + img_width * (mask_top_perc * 0.5))

    mask_points = [(mask_bottom_left_x,  bottom),
                   (mask_top_left_x,     top),
                   (mask_top_right_x,    top), 
                   (mask_bottom_right_x, bottom)]
    
    return mask_points


def apply_mask(img, mask_points):
    '''
    apply polygon to mask image
    '''
    ignore_mask_color = 255
    mask = np.zeros_like(img)
    vertices =  np.array([mask_points], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    result = cv2.bitwise_and(img, mask)
    return result


def perspective_reverse(img, corners_src, corners_dest, img_size):
    '''
    take and image and four src points in a rhombus along the lane lines
    dest points in a more linear quad, warp image to straighten effects
    of perspective transformation.
    '''
    
    src = np.float32(corners_src)
    
    dst = np.float32(corners_dest)
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    invM = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, invM



def undist_thresh_rev_proj(img, camera, debug=False):
    '''
    takes a BGR image, and a clibrated camera
    returns an image that has gone through our
    color thresholding and reverse projection pipeline
    '''

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_size = (img.shape[1], img.shape[0])
    warped_size = (1200, 1200)

    
    '''
    correct for camera distortion
    '''
    undistorted = camera.undistort(rgb)
    
    '''
    use HLS and RGB color thresholding
    '''
    combined = color_thresold(undistorted)
    
    '''
    make and apply polygon mask
    '''
    mask_points = make_mask(img_size, 0.6, 0.05, 0.9, 0.25)
    combined = apply_mask(combined, mask_points)
    
    src_cn = make_mask(img_size, 0.65, 0.05, 0.60, 0.1)
    dest_cn = make_mask(warped_size, 0.1, 0.0, 0.4, 0.36)
    
    warped, M, invM = perspective_reverse(combined, src_cn, dest_cn, warped_size)
    warped_color, M, invM = perspective_reverse(undistorted, src_cn, dest_cn, warped_size)
    
    if debug:
        draw_lines(rgb, src_cn, (255, 0, 0), 3)
        draw_lines(warped_color, dest_cn, (255), 3)

        display_n_images((rgb, warped_color, warped), 
                         ("orig", "rev perp", "rev perp lines"), 
                         (None, None, "gray"))

    return warped, M, invM 


