import cv2
import numpy as np
import time

#vidcap = cv2.VideoCapture('ball1.mp4')
#success,image = vidcap.read()
#success = True
#frames = 0

lower_threshold = np.array([100,50,50])
upper_threshold = np.array([130,255,255])

x_res = 1920
y_res = 1080
scale_factor = 6

x_res = int(x_res/scale_factor)
y_res = int(y_res/scale_factor)
vid = cv2.VideoCapture(0)

while True:
  #success,frame = vidcap.read()
  start_time = time.time()

  ret, frame = vid.read()

  frame_scaled = cv2.resize(frame, dsize=(x_res, y_res), interpolation=cv2.INTER_CUBIC)
  hsv = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    
  #noise reduction code
  kernel = np.ones((5, 5), np.uint8)
  mask_kernel = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  noise_reduction = cv2.blur(mask_kernel,(40,40))

  circles = cv2.HoughCircles(noise_reduction,cv2.HOUGH_GRADIENT,1,x_res,param1=50,param2=70,minRadius=0,maxRadius=200)

  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:

      cv2.circle(frame_scaled,(i[0],i[1]),i[2],(0,255,0),2)
      cv2.circle(frame_scaled,(i[0],i[1]),2,(0,0,255),3)

  cv2.imshow('circles', frame_scaled)
  cv2.imshow('what he sees',noise_reduction)

  print("FPS: ", round(1.0 / (time.time() - start_time)))

  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

  