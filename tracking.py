''' This module will be used to analyze captured video for rat motion tracking'''

import cv
from sklearn.cluster import KMeans

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
    running_average = cv.CreateImage(frame_size, cv.IPL_DEPTH_64F, 1)
    running_average_scaled = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
    
    # Greyscale image, thresholded to create the motion mask:
    grey_image = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
    post_process = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
    contour_src = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
    
    # Create an image for the difference
    difference = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)

    while frame != None:
        
        # Convert the image to greyscale.
        cv.CvtColor( frame, grey_image, cv.CV_RGB2GRAY )
        
        # Smooth that frame!
        cv.Smooth( grey_image, grey_image, cv.CV_GAUSSIAN, 9, 0)
        
        # Threhold the image to pull out the LEDs
        cv.Threshold( grey_image, grey_image, 150, 255, cv.CV_THRESH_BINARY )
        
        # This sets averaging weight
        a = 0.05
        # Use the running avg as background
        cv.RunningAvg(  grey_image, running_average, a, None )
        
        cv.ConvertScale( running_average, running_average_scaled, 1, 0)
        
        # Subtract the background from the frame
        cv.AbsDiff( running_average_scaled, grey_image, difference)
        cv.Smooth( difference, difference, cv.CV_GAUSSIAN,9, 0)
        
        # Threshold the image to a black and white motion mask:
        cv.Threshold( difference, post_process, 50, 255, cv.CV_THRESH_BINARY )
        
        # Find the contours of the LEDs
        cv.Copy(post_process, contour_src)
        
        # The idea with this next part is that we can find the contours around
        # the light blobs and they will be circles, or ovals, or peanut shapes.
        # Then if you do k-means, you should get back the centers of the blobs.
        # Find the contours around our lights
        contour_seq = cv.FindContours(contour_src, cv.CreateMemStorage(),
            mode=cv.CV_RETR_EXTERNAL, method=cv.CV_CHAIN_APPROX_NONE)
        cv.DrawContours(contour_src, contour_seq, cv.RGB(0, 0, 255), 
            cv.RGB(0, 255, 0), 1)
        
        # Fit k-means to the found contours
        if 'centers' in locals():
            # If we initialize k-means from the found centers of the previous
            # frame, hopefully it'll grab onto the light blobs and not other
            # blobs elsewhere in the frame
            km = KMeans(k = 2, init = centers)
        else:
            km = KMeans(k = 2)
        try:
            # Sometimes no contours are found, so you can't fit k-means
            km.fit(contour_seq[:])
            centers = km.cluster_centers_.astype(int)
        except:
            # So just keep the same k-means centers for this frame and maybe
            # later you can interpolate
            pass
        
        # Draw circles at the k-means centers
        cv.Circle(frame, tuple(centers[0]),10,cv.RGB(0,255,0))
        cv.Circle(frame, tuple(centers[1]),10,cv.RGB(0,0,255))
        
        # This is just showing the different steps of the analysis
        cv.ShowImage('avg viewer', contour_src)
        cv.ShowImage('pre viewer', grey_image)
        cv.ShowImage('diff viewer', difference)
        cv.ShowImage('post viewer', post_process)
        cv.ShowImage('frame', frame)
        cv.WaitKey(10)
        
        # Grab the next frame for the next loop
        frame = cv.QueryFrame(capture)
