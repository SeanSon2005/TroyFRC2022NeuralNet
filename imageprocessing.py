import cv2
import numpy as np
import time

blue_ball = True #determine ally color

#PID controller coefficients
Kp = 1 #coefficient for proportional
Ki = 0 #coefficient for integral
Kd = 0 #coefficient for derivative
Kdist = 1 #coefficient for distance

integral_previous = 0
start_time = time.time()
distance_avg = np.array([1,1,1,1,1,1])

if(blue_ball):
  lower_threshold = np.array([100,40,40])
  upper_threshold = np.array([130,255,255])
else:
  lower_threshold = np.array([0,40,40])
  upper_threshold = np.array([8,255,255])
  lower_threshold2 = np.array([170,40,40])
  upper_threshold2 = np.array([179,255,255])

scale_factor = 1

vid = cv2.VideoCapture(0)
ret, test_frame = vid.read()

x_res = int(test_frame.shape[1]/scale_factor)
y_res = int(test_frame.shape[0]/scale_factor)


def distance_estimation(radius):
    global distance_avg
    distance_avg = np.append(np.delete(np.copy(distance_avg),0),radius)
    tune = (int)(np.average(distance_avg)) - 50
    if(tune < 0):
        tune = 0
    if(tune > 40):
        tune = 40
    return tune

#PID calculations
def PIDCalc(x_value, distance):
  #declares integral previous and start time as the global ones
  global integral_previous
  global start_time

  #x_value adjustments
  x_value = 160 - x_value

  #PID calculations
  errorP = x_value * Kp
  errorI = (integral_previous + (x_value * (time.time()-start_time))) * Ki
  errorD = Kd
  error = errorP + errorI + errorD
  if (error > 0):
      error += distance*Kdist
  else:
      error -= distance*Kdist

  #updating integral
  integral_previous = errorI

  #update start time
  start_time = time.time()
  return error/120


if(blue_ball):
  while True:
    #getting video frame
    ret, frame = vid.read()
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
        print("PID",PIDCalc(i[0],distance_estimation(i[2])))
    
    cv2.imshow("result",noise_reduction)

else:
  while True:

    ret, frame = vid.read()
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
        print("sadge")

    #print("FPS: ", round(1.0 / (time.time() - start_time)))