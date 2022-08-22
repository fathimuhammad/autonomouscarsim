#!/usr/bin/env python3
from itertools import count
from operator import le
from turtle import right
from typing import Counter
import cv2
import os
import rospy
import numpy as np
import time
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
from std_msgs.msg import Float64

bridge = CvBridge()
avoidance = 0
oldtime=0
def nothing(x):
		pass

def callback(data):
    global avoidance, oldtime
    # Defining variables to hold meter-to-pixel conversion
    ym_per_pix = 30 / 800
    # Standard lane width is 3.7 meters divided by lane width in pixels which is
    # calculated to be approximately 720 pixels not to be confused with frame height
    xm_per_pix = 3.7 / 800

    velocity_publisher = rospy.Publisher('/catvehicle/cmd_vel_safe', Twist, queue_size=1)
    vel_msg = Twist()

    
    # Init Twist values
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    def move(x,z):
        vel_msg.linear.x = x
        vel_msg.angular.z = z
        velocity_publisher.publish(vel_msg)  

    def processImage(inpImage):
        hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 160, 10])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(inpImage, lower_white, upper_white)
        hls_result = cv2.bitwise_and(inpImage, inpImage, mask = mask)
        gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh,(3, 3), 0)
        canny = cv2.Canny(blur, 40, 60)
        return image, hls_result, gray, thresh, blur, canny

    def perspectiveWarp(inpImage):
        global avoidance, oldtime

        # Get image size
        img_size = (inpImage.shape[1], inpImage.shape[0])

        src = np.float32([[300, 470],
                        [785, 470],
                        [130, 605],
                        [1500, 605]])
        # Window to be shown
        dst = np.float32([[200, 0],
                        [1200, 0],
                        [200, 710],
                        [1200, 710]])

        # Matrix to warp the image for birdseye window
        matrix = cv2.getPerspectiveTransform(src, dst)
        # Inverse matrix to unwarp the image for final window
        minv = cv2.getPerspectiveTransform(dst, src)
        birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

        # Get the birdseye window dimensions
        height, width = birdseye.shape[:2]
        birdseyeLeft  = birdseye[0:height, 0:width // 2]
        birdseyeRight = birdseye[0:height, width // 2:width]
        grayImageL = cv2.cvtColor(birdseyeLeft, cv2.COLOR_BGR2GRAY)
        grayImageR = cv2.cvtColor(birdseyeRight, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImageL) = cv2.threshold(grayImageL, 127, 255, cv2.THRESH_BINARY)
        (thresh, blackAndWhiteImageR) = cv2.threshold(grayImageR, 127, 255, cv2.THRESH_BINARY)
        contoursL, hierarchy = cv2.findContours(blackAndWhiteImageL, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoursR, hierarchy = cv2.findContours(blackAndWhiteImageR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        minimum_distance = rospy.wait_for_message('/distanceEstimator/dist', Float64, timeout=10).data
        
        if minimum_distance <=20 and avoidance == 0:
            oldtime = int(time.time())
            avoidance = 1
        current_time = int(time.time())
        if current_time == oldtime + 4:
            avoidance = 0

        cv2.putText(img, "Obstacle Range = " + str(minimum_distance), (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102,255,0), 2, cv2.LINE_AA)
        if len(contoursL) > len(contoursR):
            cv2.putText(img, "Right Lane" , (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102,255,0), 2, cv2.LINE_AA)
            if avoidance == 1:
                move(3, 0.2)
        else:
            cv2.putText(img, "Left Lane" , (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102,255,0), 2, cv2.LINE_AA)
            if avoidance == 0:
                move(3, -0.25)
        # Display birdseye view image
        #cv2.imshow("Birdseye" , birdseye)
        #cv2.imshow("Birdseye Left" , birdseyeLeft)
        #cv2.imshow("Birdseye Right", birdseyeRight)

        return birdseye, birdseyeLeft, birdseyeRight, minv

    def plotHistogram(inpImage):

        histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis = 0)

        midpoint = np.int32(histogram.shape[0] / 2)
        leftxBase = np.argmax(histogram[:midpoint])
        rightxBase = np.argmax(histogram[midpoint:]) + midpoint

        plt.xlabel("Image X Coordinates")
        plt.ylabel("Number of White Pixels")

        # Return histogram and x-coordinates of left & right lanes to calculate
        # lane width in pixels
        return histogram, leftxBase, rightxBase

    def slide_window_search(binary_warped, histogram):

        # Find the start of left and right lane lines using histogram info
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        midpoint = np.int32(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # A total of 9 windows will be used
        nwindows = 9
        window_height = np.int32(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []


        #### START - Loop to iterate through windows and search for lane lines #####
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high),
            (0,255,0), 2)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds])) 
        #### END - Loop to iterate through windows and search for lane lines #######
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Apply 2nd degree polynomial fit to fit curves
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)


            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            ltx = np.trunc(left_fitx)
            rtx = np.trunc(right_fitx)
            plt.plot(right_fitx)
            # plt.show()

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # plt.imshow(out_img)
            plt.plot(left_fitx,  ploty, color = 'yellow')
            plt.plot(right_fitx, ploty, color = 'yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)

            return ploty, left_fit, right_fit, ltx, rtx
        except (TypeError, ZeroDivisionError):
            # handle multiple exceptions
            # TypeError and ZeroDivisionError
            pass

    def general_search(binary_warped, left_fit, right_fit):

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


        ## VISUALIZATION ###########################################################
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.plot(left_fitx,  ploty, color = 'yellow')
        plt.plot(right_fitx, ploty, color = 'yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        ret = {}
        ret['leftx'] = leftx
        ret['rightx'] = rightx
        ret['left_fitx'] = left_fitx
        ret['right_fitx'] = right_fitx
        ret['ploty'] = ploty

        return ret

    def measure_lane_curvature(ploty, leftx, rightx):

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Fit new polynomials to x, y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)



        # Calculate the new radii of curvature
        left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        # Decide if it is a left or a right curve
        if leftx[0] - leftx[-1] > 60:
            curve_direction = 'Left Curve'
        elif leftx[-1] - leftx[0] > 60:
            curve_direction = 'Right Curve'
        else:
            curve_direction = 'Straight'

        return (left_curverad + right_curverad) / 2.0, curve_direction

    def draw_lane_lines(original_image, warped_image, Minv, draw_info):
        leftx = draw_info['leftx']
        rightx = draw_info['rightx']
        left_fitx = draw_info['left_fitx']
        right_fitx = draw_info['right_fitx']
        ploty = draw_info['ploty']

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

        return pts_mean, result

    def offCenter(meanPts, inpFrame):
        # Calculating deviation in meters
        mpts = meanPts[-1][-1][-2].astype(int)
        pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
        deviation = pixelDeviation * xm_per_pix
        direction = "left" if deviation < 0 else "right"
        return deviation, direction

    def addText(img, radius, direction, deviation, devDirection):
        # Add the radius and center position to the image
        font = cv2.FONT_HERSHEY_SIMPLEX

        if (direction != 'Straight'):
            text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
            text1 = 'Curve Direction: ' + (direction)

        else:
            text = 'Radius of Curvature: ' + 'N/A'
            text1 = 'Curve Direction: ' + (direction)

        cv2.putText(img, text , (10,30), font, 0.8, (102,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, text1, (10,60), font, 0.8, (102,255,0), 2, cv2.LINE_AA)

        # Deviation
        deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
        cv2.putText(img, deviation_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102,255,0), 2, cv2.LINE_AA)

        return img

    # Read the input image
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    frame = img
    image = frame
    #### START - LOOP TO PLAY THE INPUT IMAGE ######################################
    # Apply perspective warping by calling the "perspectiveWarp()" function
    # Then assign it to the variable called (birdView)
    # Provide this function with:
    # 1- an image to apply perspective warping (frame)
    birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)


    # Apply image processing by calling the "processImage()" function
    # Then assign their respective variables (img, hls, grayscale, thresh, blur, canny)
    # Provide this function with:
    # 1- an already perspective warped image to process (birdView)
    img, hls, grayscale, thresh, blur, canny = processImage(birdView)
    imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
    imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)


    # Plot and display the histogram by calling the "get_histogram()" function
    # Provide this function with:
    # 1- an image to calculate histogram on (thresh)
    hist, leftBase, rightBase = plotHistogram(thresh)
    # print(rightBase - leftBase)
    plt.plot(hist)
    # plt.show()

    try:
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)
        plt.plot(left_fit)
        # plt.show()
        draw_info = general_search(thresh, left_fit, right_fit)
        # plt.show()

        curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)
    except (TypeError, ZeroDivisionError):
            pass
    
    # Filling the area of detected lanes with green
    meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)

    deviation, directionDev = offCenter(meanPts, frame)

    # Adding text to our final image
    finalImg = addText(result, curveRad, curveDir, deviation, directionDev)
    # Displaying final image
    cv2.imshow("Cam", finalImg)
    speed = 8
    speed = speed - (abs(deviation)*4)
    if speed <= 5:
        speed = 5
    factor = 0.07
    move(speed,(deviation*abs(deviation))*factor)
    cv2.waitKey(1)

def receive():
    rospy.Subscriber("/catvehicle/camera_front/image_raw_front", Image, callback)
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node("receiveImage" , anonymous=True)
    try:
        receive()
    except rospy.ROSInterruptException: pass