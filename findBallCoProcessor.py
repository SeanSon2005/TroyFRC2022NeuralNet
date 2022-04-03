import cv2
import numpy as np
import time
from cscore import CameraServer
from networktables import NetworkTable, NetworkTables
import threading
import math

blue_ball = True #determine ally color Blue(True) or Red(False)
testing_on_computer = False #testing on roborio(False) or computer(True)

#PID controller coefficients
Kp = 1 #coefficient for proportional
Ki = 0.2 #coefficient for integral
Kd = 0 #coefficient for derivative

integral_previous = 0
start_time = time.time()

if(blue_ball):
  lower_threshold = np.array([100,40,40])
  upper_threshold = np.array([130,255,255])
else:
  lower_threshold = np.array([0,40,40])
  upper_threshold = np.array([8,255,255])
  lower_threshold2 = np.array([170,40,40])
  upper_threshold2 = np.array([179,255,255])

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


if(blue_ball):
  while True:
    #starting time for finding Frames per second
    #start_time = time.time()

    #getting video frame
    ret, frame_scaled = cvSink.grabFrame(img)
  
    #convert image to HSV
    hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)

    #mask image with color range (blue)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
      
    #noise reduction code (also reduimentary "contours")
    noise_reduction = cv2.blur(mask,(15,15))
    noise_reduction = cv2.inRange(noise_reduction,1,70)
    noise_reduction = cv2.blur(noise_reduction,(15,15))

    #find Circles
    circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,minDist=25,param1=50,param2=70,minRadius=10,maxRadius=120)

    if circles is not None:
      circles = np.uint16(np.around(circles))
      max_radius = math.floor((circles.argmax()+1)/3)
      x_pos = circles[0,max_radius,0]
      #print(PIDCalc(x_pos))
      vision_nt.putNumber('PID',PIDCalc(x_pos))
    
    #Print Frames Per Second
    #print("FPS: ", round(1.0 / (time.time() - start_time)))

else:
  while True:
    #success,frame = vidcap.read()
    start_time = time.time()

    ret, frame = cvSink.grabFrame(img)
    frame_scaled = cv2.resize(frame, dsize=(x_res, y_res), interpolation=cv2.INTER_CUBIC)
    
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
      circles = np.uint16(np.around(circles))
      max_radius = math.floor((circles.argmax()+1)/3)
      x_pos = circles[0,max_radius,0]
      #print(PIDCalc(x_pos))
      vision_nt.putNumber('PID',PIDCalc(x_pos))

    #print("FPS: ", round(1.0 / (time.time() - start_time)))