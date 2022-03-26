import cv2
import numpy as np
import time

blue_ball = True

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

if(blue_ball):
  while True:
    #success,frame = vidcap.read()
    start_time = time.time()

    ret, frame = vid.read()


    frame_scaled = cv2.resize(frame, dsize=(x_res, y_res), interpolation=cv2.INTER_CUBIC)
    
    hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)
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

        cv2.circle(frame_scaled,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(frame_scaled,(i[0],i[1]),2,(0,0,255),3)

    #cv2.imshow('circles', frame_scaled)
    #cv2.imshow('what he sees',mask)

    print("FPS: ", round(1.0 / (time.time() - start_time)))

    #if cv2.waitKey(1) & 0xFF == ord('q'):
          #break
else:
  while True:
    #success,frame = vidcap.read()
    start_time = time.time()

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

        cv2.circle(frame_scaled,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(frame_scaled,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('circles', frame_scaled)
    cv2.imshow('what he sees',mask)

    print("FPS: ", round(1.0 / (time.time() - start_time)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
          break


vid.release()
cv2.destroyAllWindows()