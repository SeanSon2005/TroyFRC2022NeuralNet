import cv2
from cv2 import contourArea
import numpy as np
import time
import math

blue_ball = True #determine ally color

#PID controller coefficients
Kp = 1 #coefficient for proportional
Ki = 0.2 #coefficient for integral
Kd = 0 #coefficient for derivative

integral_previous = 0
start_time = time.time()
distance_avg = np.array([1,1,1,1,1,1])

if(blue_ball):
  lower_threshold = np.array([100,40,25])
  upper_threshold = np.array([130,255,255])
else:
  lower_threshold = np.array([0,40,40])
  upper_threshold = np.array([8,255,255])
  lower_threshold2 = np.array([170,40,40])
  upper_threshold2 = np.array([179,255,255])

#PID calculations
def PIDCalc(x_value):
  #declares integral previous and start time as the global ones
  global integral_previous
  global start_time

  #x_value adjustments
  x_value = 320 - x_value

  #PID calculations
  errorP = x_value * Kp
  errorI = (integral_previous + (x_value * (time.time()-start_time))) * Ki
  errorD = Kd
  error = errorP + errorI + errorD

  #updating integral
  integral_previous = errorI

  #update start time
  start_time = time.time()
  return error/120



scale_factor = 1

vid = cv2.VideoCapture(0)
ret, test_frame = vid.read()

x_res = int(test_frame.shape[1]/scale_factor)
y_res = int(test_frame.shape[0]/scale_factor)

if(blue_ball):
  while True:
    #getting video frame
    start_time = time.time()

    ret, frame = vid.read()
    frame_scaled = cv2.resize(frame, dsize=(x_res, y_res), interpolation=cv2.INTER_CUBIC)
    
    blurred_original = cv2.blur(frame_scaled,(4,4))
    #convert image to HSV
    hsv = cv2.cvtColor(blurred_original, cv2.COLOR_BGR2HSV)

    #mask image with color range (blue)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)


    #find Edges
    contours, hierachry = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = cv2.drawContours(np.zeros(frame_scaled.shape, dtype='uint8'), contours, contourIdx= -1, color = (255,255,255), thickness=2)
    grayImage = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.blur(grayImage,(15,15))

    circles = cv2.HoughCircles(grayImage,cv2.HOUGH_GRADIENT,minDist=15,dp=1.5,param1=1,param2=100,minRadius=15,maxRadius=130)

    if circles is not None:
      circles = np.uint16(np.around(circles))
      max_radius = math.floor((circles.argmax()+1)/3)
      x_pos = circles[0,max_radius,0]
      y_pos = circles[0,max_radius,1]
      print("x:",x_pos,"y:",y_pos)
      for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(frame_scaled,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(frame_scaled,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('Objects Detected',frame_scaled)
    cv2.imshow("normal",grayImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    #print("FPS: ", round(1.0 / (time.time() - start_time)))

vid.release()
cv2.destroyAllWindows()