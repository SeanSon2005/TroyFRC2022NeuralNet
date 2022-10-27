import cv2
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
frame_count = 0
frame_num = 0

if(blue_ball):
  lower_threshold = np.array([100,40,40])
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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open usb camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


ret, test_frame = cap.read()

x_res = int(test_frame.shape[1]/scale_factor)
y_res = int(test_frame.shape[0]/scale_factor)

if(blue_ball):
  while True:
    start_time = time.time()

    #getting video frame
    ret, frame = cap.read()
    
    #convert image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #mask image with color range (blue)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
      
    #noise reduction code
    noise_reduction = cv2.blur(mask,(10,10))
    noise_reduction = cv2.inRange(noise_reduction,1,75)
    noise_reduction = cv2.blur(noise_reduction,(15,15))

    circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,minDist=25,param1=50,param2=70,minRadius=10,maxRadius=120)


    
    #cv2.imshow("result",noise_reduction)
    cv2.imshow("normal",frame)
    #cv2.imshow("normal2",noise_reduction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    fps = round(1.0 / (time.time() - start_time))
    print("FPS: ", fps)

    frame_count += fps
    frame_num += 1

else:
  while True:

    ret, frame = cap.read()
    frame_scaled = cv2.resize(frame, dsize=(x_res, y_res), interpolation=cv2.INTER_CUBIC)
    
    hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_threshold, upper_threshold)
    mask2 = cv2.inRange(hsv, lower_threshold2, upper_threshold2)
    mask = mask1 + mask2
      
    #noise reduction code
    noise_reduction = cv2.blur(mask,(15,15))
    noise_reduction = cv2.inRange(noise_reduction,10,70)
    noise_reduction = cv2.blur(noise_reduction,(15,15)) 

    circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1.3,x_res,param1=50,param2=70,minRadius=1,maxRadius=120)

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

    #print("FPS: ", round(1.0 / (time.time() - start_time)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
print("avg FPS",(frame_count/frame_num))