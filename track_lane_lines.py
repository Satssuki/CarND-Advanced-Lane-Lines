import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import camera
import lane_lines
import image_process
import pickle
from moviepy.editor import VideoFileClip

class LineProcessor():
    def __init__(self, cam):
        self.cam = cam

    def process(self, image):
        '''
        take as input the image
        returns processed image
        '''
        result = process_lane_lines(image, self.cam)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)       

        return result


def get_camera(camera_pickle_filename):
    '''
    attempt to load a previous calibrated camera,
    failing that, calibrate a new one and save it to
    the pickle file
    '''
    try:
        with open(camera_pickle_filename, 'rb') as infile:
            camera = pickle.load(infile)
    except:
        camera = camera.Camera()

        nx = 9 #The number of inside corners in x
        ny = 6 #The number of inside corners in y

        camera.calib('camera_cal/*.jpg', nx, ny)

        with open(camera_pickle_filename, 'wb') as outfile:
            pickle.dump(camera, outfile)

    return camera


def process_lane_lines(img, camera):
    orig_image_shape = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    '''
    do color thresh and reverse perspective transform
    '''
    warped, M, invM = image_process.undist_thresh_rev_proj(img, camera, debug=False)


    '''
    fit the left and right polynomial lines
    '''
    left_lane, right_lane = lane_lines.discover_two_lines(warped)


    '''
    create an image overlay of lines
    '''
    newwarp = lane_lines.get_lane_overlay(warped, 
                left_lane, 
                right_lane, 
                invM, 
                orig_image_shape)
    

    '''
    Combine the result with the original image
    '''
    result = cv2.addWeighted(rgb, 1, newwarp, 0.3, 0)

    '''
    write the lane position and curvature over image
    '''
    lane_lines.write_lane_info(left_lane, 
                right_lane,
                result,
                warped.shape)
                
    
    return result

def process_test_images():
    files = glob.glob('test_images/*.jpg')
    camera_pickle_filename = 'camera.pkl'

    camera = get_camera(camera_pickle_filename)


    for infile in files:
        print('processing', infile)
        
        img = cv2.imread(infile)

        result = process_lane_lines(img, camera)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        outfilename = infile.replace('test_images', 'output_images')

        print('writing', outfilename)

        cv2.imwrite(outfilename, result)


def process_movie(infilename, outfilename):  

    camera_pickle_filename = 'camera.pkl'

    camera = get_camera(camera_pickle_filename)

    
    clip = VideoFileClip(infilename)

    line_processor = LineProcessor(camera)

    out_clip = clip.fl_image(line_processor.process)

    out_clip.write_videofile(outfilename, audio=False)



process_test_images()

#process_movie("project_video.mp4", "project_video_out.mp4")
