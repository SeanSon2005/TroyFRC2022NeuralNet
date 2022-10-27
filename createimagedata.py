
import cv2
import numpy as np
import time
from PIL import Image
import os
import keyboard 

path = "C://Users//astro//Documents//Coding//yolov7//train"
os.chdir(path+"//images")

scale_factor = 1
camera_number = 1

cap = cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open usb camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
    ret, frame = cap.read()
    
    cv2.imshow("res",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if keyboard.is_pressed('space'):
        file_count = len(os.listdir())
        im = Image.fromarray(frame)
        im.save("image_"+str(file_count)+".png")
        print("Captured")

cap.release()
cv2.destroyAllWindows()