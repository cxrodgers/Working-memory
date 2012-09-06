''' This module will be used to analyze captured video for rat motion tracking'''

import cv
from sklearn.cluster import KMeans

#~ # Path to the video file
#~ file_path = ''

#~ # This grabs the video file and returns an object that you can grab frames from
#~ video = cv.CaptureFromFile(file_path)

#~ # The number of frames in the video and the frame rate
#~ nFrames = cv.GetCaptureProperty(cap, 7)
#~ frame_rate = cv.GetCaptureProperty(cap, 5)

def detect_motion(capture):
    
    nFrames = cv.GetCaptureProperty(capture, 7)
    frame_rate = cv.GetCaptureProperty(capture, 5)
    
    # This grabs the first frame from the video
    frame = cv.QueryFrame(capture)
    
    # Grab the parameters of the video
    frame_size = cv.GetSize(frame)
    frame_depth = frame.depth
    frame_channels = frame.nChannels
    
    # Create a running average holder
    running_average = cv.CreateImage(frame_size, cv.IPL_DEPTH_32F, 1)
    running_average_scaled = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
    
    # Greyscale image, thresholded to create the motion mask:
    grey_image = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
    grey_image_test = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
    
    # Create an image for the difference
    difference = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
    
    # Create a window so I can see what's going on
    cv.NamedWindow('avg viewer')
    cv.NamedWindow('pre viewer')
    cv.NamedWindow('thresh viewer')
    cv.NamedWindow('diff viewer')
    
    #motion = []

    while frame != None:
        
        # Convert the image to greyscale.
        cv.CvtColor( frame, grey_image_test, cv.CV_RGB2GRAY )
        
        cv.Threshold( grey_image_test, grey_image_test, 220, 255, cv.CV_THRESH_BINARY )
        
        # Smooth that frame!
        cv.Smooth( grey_image_test, grey_image_test, cv.CV_GAUSSIAN, 3, 0)
        
        # This sets averaging weight
        a = 0.02
        cv.RunningAvg(  grey_image_test, running_average, a, None )
        
        # Convert the scale of the running average
        cv.ConvertScale( running_average, running_average_scaled, 1, 0)
        
        # Subtract the current frame from the moving average
        cv.AbsDiff( running_average_scaled, grey_image_test, difference)
        
        # Smooth that frame!
        cv.Smooth( difference, difference, cv.CV_GAUSSIAN, 5, 0)
        # Convert the image to greyscale.
        #cv.CvtColor( difference, grey_image_test, cv.CV_RGB2GRAY )

        # Threshold the image to a black and white motion mask:
        cv.Threshold( difference, grey_image, 100, 255, cv.CV_THRESH_BINARY )
        
        # Smooth and threshold again to eliminate "sparkles"
        #cv.Smooth( grey_image, grey_image, cv.CV_GAUSSIAN, 3, 0 )
        #cv.Threshold( grey_image, grey_image, 2, 255, cv.CV_THRESH_BINARY )
        
        # Store the difference image
        #motion.append(difference)
        
        cv.ShowImage('avg viewer', running_average_scaled)
        cv.ShowImage('pre viewer', grey_image_test)
        cv.ShowImage('diff viewer', difference)
        cv.ShowImage('thresh viewer', grey_image)
        cv.WaitKey(10)
        
        # Grab the next frame for the next loop
        frame = cv.QueryFrame(capture)
