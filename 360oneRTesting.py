
import cv2
import numpy as np
import time

scale_factor = 1
camera_number = 1

cap = cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open usb camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
    start_time = time.time()
    ret, frame = cap.read()
    
    cv2.imshow("res",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    print("FPS: ", round(1.0 / (time.time() - start_time)))

cap.release()
cv2.destroyAllWindows()