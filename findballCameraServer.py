import cv2
import numpy as np
import time
from cscore import CameraServer
from networktables import NetworkTable, NetworkTables
import threading

blue_ball = True #determine ally color
testing_on_computer = False #testing on roborio or computer

#PID controller coefficients
Kp = 1 #coefficient for proportional
Ki = 0.1 #coefficient for integral
Kd = 0 #coefficient for derivative

integral_previous = 1
print(integral_previous)
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
img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

ret, test_frame = cvSink.grabFrame(img)

x_res = int(test_frame.shape[1]/scale_factor)
y_res = int(test_frame.shape[0]/scale_factor)

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

  #PID calculations
  x_value = 200 - x_value
  errorP = x_value * Kp
  errorI = (integral_previous + (x_value * (time.time()-start_time))) * Ki
  errorD = Kd
  error = errorP + errorI + errorD

  #updating integral
  integral_previous = errorI

  #update start time
  start_time = time.time()
  
  return error/160


if(blue_ball):
  while True:
    #getting video frame
    ret, frame = cvSink.grabFrame(img)
    frame_scaled = cv2.resize(frame, dsize=(x_res, y_res), interpolation=cv2.INTER_CUBIC)
    
    #convert image to HSV
    hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)

    #mask image with color range (blue)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
      
    #noise reduction code
    kernel = np.ones((3, 3), np.uint8)
    mask_kernel = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    noise_reduction = cv2.blur(mask,(50,50))
    noise_reduction = cv2.inRange(noise_reduction,10,50)
    noise_reduction = cv2.blur(noise_reduction,(15,15))

    circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,x_res,param1=50,param2=70,minRadius=1,maxRadius=120)

    if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0,:]:
        #print("PID",PIDCalc(i[0]))
        vision_nt.putNumber('PID',PIDCalc(i[0]))

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
      
    #noise reduction code
    kernel = np.ones((3, 3), np.uint8)
    mask_kernel = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    noise_reduction = cv2.blur(mask,(50,50))
    noise_reduction = cv2.inRange(noise_reduction,10,50)
    noise_reduction = cv2.blur(noise_reduction,(15,15))

    circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,x_res,param1=50,param2=70,minRadius=1,maxRadius=120)

    if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0,:]:
        vision_nt.putNumber('PID',PIDCalc(i[0]))

    #print("FPS: ", round(1.0 / (time.time() - start_time)))
