import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Camera(object):
    def __init__(self):
        #the calibrated camera intrisic matrix
        self.mtx = None 
        #the calibrated distance coeffecients
        self.dist = None

    def undistort(self, img):
        '''
        Takes an image and camera maxtrix and distance coeffcients
        returns the undistorted image
        '''
        new_image = cv2.undistort(src=img, cameraMatrix=self.mtx, distCoeffs=self.dist)
        
        return new_image
        
    def get_corners(self, fname, nx, ny, criteria):
        '''
        us cv2 to find the corners of chess board pattern
        '''
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        return ret, corners

    def display_corners(self, fname, nx, ny, corners):
        '''
        display a plot of the image with a color overlay of the found corners
        '''
        img = cv2.imread(fname)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, True)

        plt.title(fname)
        plt.imshow(img)
        
    def calib(self, image_path, nx, ny, show_imgs = False):
        '''
        take an image_path which is a mask to collect images
        nx is the number of expected corners along X axis
        ny is the number of expected corners along Y axis
        show_imgs will tell us whether to display each image we calib      
        returns a camera mtx
        '''
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Make a list of calibration images
        cal_files = glob.glob(image_path)
        
        img = cv2.imread(cal_files[0])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        imgpoints =[]
        objpoints = []

        if show_imgs:
            plt.rcdefaults()
            fig = plt.figure(figsize=(10,30))

        n = 0
        for fname in cal_files:
            ret, corners = self.get_corners(fname, nx, ny, criteria)
            
            if show_imgs:
                plt.subplot(10,2, n+1)
                
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                if show_imgs:
                    self.display_corners(fname, nx, ny, corners)
            else:
                if show_imgs:
                    img = cv2.imread(fname)
                    plt.title('failed:' + fname)
                    plt.imshow(img)

            n += 1

        if show_imgs:
            plt.show()
        
        dim = gray.shape[::-1]
        
        err, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                                    objectPoints=objpoints, 
                                    imagePoints=imgpoints, 
                                    imageSize=dim, 
                                    cameraMatrix=None, 
                                    distCoeffs=None)
        
        print('re-projection error', err)

        self.mtx = mtx
        self.dist = dist
        return err
