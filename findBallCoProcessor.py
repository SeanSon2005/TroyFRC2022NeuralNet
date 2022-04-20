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

integral_previous = 0
start_time = time.time()

scale_factor = 1

cs = CameraServer.getInstance()
cs.enableLogging()

camera = cs.startAutomaticCapture()

cvSink = cs.getVideo()

x_res = 640
y_res = 360

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
        lower_threshold = np.array([100,40,40])
        upper_threshold = np.array([140,255,255])
        mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    else:
        lower_threshold = np.array([0,40,40])
        upper_threshold = np.array([8,255,255])
        lower_threshold2 = np.array([170,40,40])
        upper_threshold2 = np.array([179,255,255])
        hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_threshold, upper_threshold)
        mask2 = cv2.inRange(hsv, lower_threshold2, upper_threshold2)
        mask = mask1 + mask2
      
    #noise reduction code (also reduimentary "contours")
    noise_reduction = cv2.blur(mask,(15,15))
    noise_reduction = cv2.inRange(noise_reduction,1,70)
    noise_reduction = cv2.blur(noise_reduction,(15,15))

    #find Circles
    circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,minDist=25,param1=50,param2=70,minRadius=10,maxRadius=120)

    if circles is not None:
      vision_nt.putBoolean('seeBall',True)
      circles = np.uint16(np.around(circles))
      max_radius = math.floor((circles.argmax()+1)/3)
      x_pos = circles[0,max_radius,0]
      #print(PIDCalc(x_pos))
      vision_nt.putNumber('PID',PIDCalc(x_pos))
    else:
      vision_nt.putBoolean('seeBall',False)