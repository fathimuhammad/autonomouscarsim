#!/usr/bin/env python3
from itertools import count
from operator import le
from subprocess import call
from telnetlib import Telnet
from tkinter.tix import Tree
from turtle import right
from typing import Counter
import cv2
import os
import rospy
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
from std_msgs.msg import Float64

bridge = CvBridge()
avoidance = 0
oldtime = 0
steering_angle = 0
speed = 0
minimum_distance = 80
start_time = time.time()
start_ros_time = 0
status_ros = False
duration_ros_time = 0
display_time = 2
track_detected = True
lane_left = False
fc = 0
FPS = 0


def callback0(data):
    global minimum_distance
    raw_data = str(data)
    minimum_distance = float(raw_data[6:None])


def callback(data):
    global avoidance, oldtime, status_ros, start_ros_time, steering_angle, speed, lane_left, track_detected, start_time, display_time, fc, FPS
    if status_ros == False:
        start_ros_time = rospy.get_rostime()
        status_ros = True
    ym_per_pix = 30 / 800
    xm_per_pix = 3.7 / 800
    velocity_publisher = rospy.Publisher("/catvehicle/cmd_vel_safe", Twist, queue_size=1)
    vel_msg = Twist()
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    def move(x, z):
        vel_msg.linear.x = x
        vel_msg.angular.z = z
        velocity_publisher.publish(vel_msg)

    def obstacle():
        cv2.putText(img, "OBSTACLE DETECTED !", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

    def processImage(inpImage):
        hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
        lower_white = np.array([100, 100, 100])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(inpImage, lower_white, upper_white)
        hls_result = cv2.bitwise_and(inpImage, inpImage, mask=mask)
        gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        canny = cv2.Canny(blur, 40, 60)
        return image, hls_result, gray, thresh, blur, canny

    def perspectiveWarp(inpImage):
        global avoidance, oldtime, steering_angle, lane_left, speed
        img_size = (inpImage.shape[1], inpImage.shape[0])
        src = np.float32([[315, 455], [785, 455], [130, 605], [1500, 605]])
        dst = np.float32([[200, 0], [1200, 0], [200, 710], [1200, 710]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        minv = cv2.getPerspectiveTransform(dst, src)
        birdseye = cv2.warpPerspective(inpImage, matrix, img_size)
        lower_white = np.array([100, 100, 100])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(birdseye, lower_white, upper_white)
        birdseye = cv2.bitwise_and(birdseye, birdseye, mask=mask)
        height, width = birdseye.shape[:2]
        birdseyeLeft = birdseye[0:height, 0 : width // 2]
        birdseyeRight = birdseye[0:height, width // 2 : width]
        grayImageL = cv2.cvtColor(birdseyeLeft, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImageL) = cv2.threshold(grayImageL, 127, 255, cv2.THRESH_BINARY)
        contoursL, hierarchy = cv2.findContours(blackAndWhiteImageL, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if minimum_distance <= 20 and avoidance == 0:
            oldtime = int(time.time())
            avoidance = 1
        current_time = int(time.time())
        if current_time >= oldtime + 4:
            avoidance = 2
        cv2.putText(
            img,
            "Obstacle Range : " + str(int(minimum_distance)),
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (102, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if len(contoursL) ** 2 >= 16:
            lane_left = False
            cv2.putText(
                img, "Position : Right Lane", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 255, 0), 2, cv2.LINE_AA
            )
            if avoidance == 1:
                steering_angle = 1
                obstacle()
            else:
                steering_angle = 0
        elif len(contoursL) ** 2 <= 9:
            lane_left = True
            cv2.putText(
                img, "Position : Left Lane", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 255, 0), 2, cv2.LINE_AA
            )
            if minimum_distance <= 20:
                steering_angle = -1
                obstacle()
            if avoidance == 2:
                steering_angle = -1
                avoidance = 0
            else:
                steering_angle = 0
        return birdseye, birdseyeLeft, birdseyeRight, minv

    def plotHistogram(inpImage):
        histogram = np.sum(inpImage[inpImage.shape[0] // 2 :, :], axis=0)
        midpoint = np.int32(histogram.shape[0] / 2)
        leftxBase = np.argmax(histogram[:midpoint])
        rightxBase = np.argmax(histogram[midpoint:]) + midpoint
        return histogram, leftxBase, rightxBase

    def slide_window_search(binary_warped, histogram):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        midpoint = np.int32(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
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
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        ltx = np.trunc(left_fitx)
        rtx = np.trunc(right_fitx)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return ploty, left_fit, right_fit, ltx, rtx

    def general_search(binary_warped, left_fit, right_fit):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)
        ) & (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin))
        right_lane_inds = (
            nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)
        ) & (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        ret = {}
        ret["leftx"] = leftx
        ret["rightx"] = rightx
        ret["left_fitx"] = left_fitx
        ret["right_fitx"] = right_fitx
        ret["ploty"] = ploty
        return ret

    def measure_lane_curvature(ploty, leftx, rightx):
        leftx = leftx[::-1]
        rightx = rightx[::-1]
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0]
        )
        right_curverad = (
            (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
        ) / np.absolute(2 * right_fit_cr[0])
        if leftx[0] - leftx[-1] > 60:
            curve_direction = "Left Curve"
        elif leftx[-1] - leftx[0] > 60:
            curve_direction = "Right Curve"
        else:
            curve_direction = "Straight"
        return (left_curverad + right_curverad) / 2.0, curve_direction

    def draw_lane_lines(original_image, warped_image, Minv, draw_info):
        leftx = draw_info["leftx"]
        rightx = draw_info["rightx"]
        left_fitx = draw_info["left_fitx"]
        right_fitx = draw_info["right_fitx"]
        ploty = draw_info["ploty"]
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
        mpts = meanPts[-1][-1][-2].astype(int)
        pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
        deviation = pixelDeviation * xm_per_pix
        direction = "left" if deviation < 0 else "right"
        return deviation, direction

    def addText(img, radius, direction, deviation, devDirection):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if direction != "Straight":
            text = "Radius of Curvature : " + "{:04.0f}".format(radius) + "m"
            text1 = "Curve Direction : " + (direction)
        else:
            text = "Radius of Curvature : " + "N/A"
            text1 = "Curve Direction : " + (direction)
        cv2.putText(img, text, (10, 30), font, 0.8, (102, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text1, (10, 60), font, 0.8, (102, 255, 0), 2, cv2.LINE_AA)
        deviation_text = "Off Center: " + str(round(abs(deviation), 3)) + "m" + " to the " + devDirection
        cv2.putText(img, deviation_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 255, 0), 2, cv2.LINE_AA)
        return img

    img = bridge.imgmsg_to_cv2(data, "bgr8")
    frame = img
    image = frame
    fc += 1
    TIME = time.time() - start_time
    if (TIME) >= display_time:
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()
    fps_disp = "FPS : " + str(FPS)[:5]
    if int(FPS) <= 17:
        cv2.putText(img, "LOW FPS !!!", (320, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(img, str(fps_disp), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, str(fps_disp), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 255, 0), 2, cv2.LINE_AA)
    birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)
    img, hls, grayscale, thresh, blur, canny = processImage(birdView)
    imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
    imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)
    hist, leftBase, rightBase = plotHistogram(thresh)
    try:
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)
        draw_info = general_search(thresh, left_fit, right_fit)
        curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)
        meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)
        deviation, directionDev = offCenter(meanPts, frame)
        finalImg = addText(result, curveRad, curveDir, deviation, directionDev)
        track_detected = True
    except:
        deviation = 0
        curveRad = 0
        directionDev = 0
        finalImg = image
        track_detected = False
        pass
    if lane_left == True:
        speed = 6.75
    else:
        speed = 11
    factor = 0.09
    speed = speed - abs(deviation) ** 2
    if curveRad <= 400:
        speed = 7.6
        factor = 0.06
    elif curveRad <= 450:
        speed = 8.6
        factor = 0.07   
    elif curveRad <= 500:
        speed = 9
        factor = 0.08
        
    if abs(deviation) <= 0.1:
        speed = speed + 2
        
    if speed <= 3:
        speed = 3
    if track_detected == False:
        speed = 3
    else:
        track_detected = True
    move(speed, steering_angle + (deviation * factor))
    cv2.putText(
        finalImg,
        "Speed : " + str(int(speed * 10)) + " Km/H",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (102, 255, 0),
        2,
        cv2.LINE_AA,
    )
    duration_ros_time = rospy.get_rostime() - start_ros_time
    cv2.putText(
        finalImg,
        "Duration : " + format(duration_ros_time.to_sec(),'.2f') + " Sec",
        (10, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (102, 255, 0),
        2,
        cv2.LINE_AA,
    )
    # cv2.imshow("Camera View", img)
    cv2.imshow("Camera View", finalImg)
    cv2.waitKey(1)


def receive():
    rospy.Subscriber("/distanceEstimator/dist", Float64, callback0)
    rospy.Subscriber("/catvehicle/camera_front/image_raw_front", Image, callback)
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("receiveImage", anonymous=True)
    try:
        receive()
    except rospy.ROSInterruptException:
        pass
