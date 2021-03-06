import cv2
import numpy as np
import time
from cscore import CameraServer
from networktables import NetworkTable, NetworkTables
import threading
import math

testing_on_computer = False #testing on roborio(False) or computer(True)

#PID controller coefficients
Kp = 1 #coefficient for proportional
Ki = 0.2 #coefficient for integral
Kd = 0 #coefficient for derivative

x_res = 640 #camera width
y_res = 360 #camera height

integral_previous = 0
start_time = time.time()

#Camera Setup
cs = CameraServer.getInstance()
cs.enableLogging()

camera = cs.startAutomaticCapture()
cvSink = cs.getVideo()
output = cs.putVideo("Front Camera", x_res, y_res)

#declaring thresholds for both red and blue balls
lower_thresholdRED = np.array([0,40,40])
upper_thresholdRED = np.array([8,255,255])
lower_threshold2RED = np.array([170,40,40])
upper_threshold2RED = np.array([179,255,255])
lower_threshold = np.array([100,140,65])
upper_threshold = np.array([110,225,215])

img = np.zeros(shape=(x_res, y_res, 3), dtype=np.uint8)

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    #print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

#only runs if running through Roborio (boolean set above)
if not testing_on_computer:
  NetworkTables.initialize(server='10.39.52.2')
  NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

  #waits for Network Tables
  with cond:
      print("Waiting")
      if not notified[0]:
          cond.wait()
  #get the table
  vision_nt = NetworkTables.getTable('Vision')

#PID calculations
def PIDCalc(x_value):
  #declares integral previous and start time as the global ones
  global integral_previous
  global start_time

  #x_value adjustments (find error relative to center: 320p)
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
  
  return error/320


if not testing_on_computer:
  while True:
      #starting time for finding Frames per second
      #start_time = time.time()

      #getting video frame
      ret, frame_scaled = cvSink.grabFrame(img)
    
      #convert image to HSV
      hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)

      #finds out team color through network tables
      blue_ball = vision_nt.getBoolean("blueBall", True)

      if(blue_ball):
          #mask image with color range (blue)
          mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
      else:
          #mask image with color range (red)
          hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)
          mask1 = cv2.inRange(hsv, lower_thresholdRED, upper_thresholdRED)
          mask2 = cv2.inRange(hsv, lower_threshold2RED, upper_threshold2RED)
          mask = mask1 + mask2
        
      #noise reduction code (also reduimentary "contours")
      noise_reduction = cv2.blur(mask,(15,15))
      noise_reduction = cv2.inRange(noise_reduction,1,70)
      noise_reduction = cv2.blur(noise_reduction,(15,15))

      #find Circles
      #param1 canny edge parameter
      #param2 the strictness of circle detection
      #minRadius the minimum radius for a circle detection
      #maxRadius the maximum radius for a circle detection
      circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,minDist=25,param1=50,param2=70,minRadius=10,maxRadius=120)

      if circles is not None:
        vision_nt.putBoolean('seeBall',True)
        circles = np.uint16(np.around(circles))[0]
        index = np.argmax(circles[:,2])
        x_pos = circles[index][0]
        y_pos = circles[index][1]
        pidVal = PIDCalc(x_pos)
        cv2.circle(frame_scaled,(x_pos,y_pos),5,(0,255,0),2)
        cv2.putText(img = frame_scaled,text = str(pidVal),org = (10, 340),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 2)
        vision_nt.putNumber('PID',pidVal)
      else:
        vision_nt.putBoolean('seeBall',False)

      output.putFrame(frame_scaled) #puts frame into Camera Server
else:
  while True:
      #starting time for finding Frames per second
      #start_time = time.time()

      #getting video frame
      ret, frame_scaled = cvSink.grabFrame(img)
    
      #convert image to HSV
      hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)

      #mask image with color range (blue)
      hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)
      mask1 = cv2.inRange(hsv, lower_thresholdRED, upper_thresholdRED)
      mask2 = cv2.inRange(hsv, lower_threshold2RED, upper_threshold2RED)
      mask = mask1 + mask2
        
      #noise reduction code (also reduimentary "contours")
      noise_reduction = cv2.blur(mask,(15,15))
      noise_reduction = cv2.inRange(noise_reduction,1,70)
      noise_reduction = cv2.blur(noise_reduction,(15,15))

      #find Circles
      #param1 canny edge parameter
      #param2 the strictness of circle detection
      #minRadius the minimum radius for a circle detection
      #maxRadius the maximum radius for a circle detection
      circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,minDist=25,param1=50,param2=70,minRadius=10,maxRadius=120)

      if circles is not None:
        #vision_nt.putBoolean('seeBall',True)
        circles = np.uint16(np.around(circles))[0]
        index = np.argmax(circles[:,2])
        x_pos = circles[index][0]
        y_pos = circles[index][1]
        cv2.circle(frame_scaled,(x_pos,y_pos),5,(0,255,0),2)
        cv2.putText(frame_scaled,text = str(PIDCalc(x_pos)),org = (10, 340),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 2)
      output.putFrame(frame_scaled) #puts frame into Camera Server