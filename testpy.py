import cv2
import numpy as np
from cscore import CameraServer, VideoMode, UsbCamera, VideoSink
import os

x_res = 1280 #camera width
y_res = 720 #camera height

cs = CameraServer.getInstance()
cs.enableLogging()

#print(os.)
camera = UsbCamera("Camera", '/dev/video0')
camera.setResolution(x_res, y_res)
camera.setPixelFormat(VideoMode.PixelFormat.kMJPEG)
camera.setFPS(30)
cs.startAutomaticCapture(camera=camera)

cvSink = cs.getVideo(camera=camera)
outputStream = cs.putVideo("Camera Camera", x_res, y_res)

img = np.zeros(shape=(x_res, y_res, 3), dtype=np.uint8)

while True:
    ret,img = cvSink.grabFrame(img)
    if ret == 0:
        outputStream.notifyError(cvSink.getError())
        continue
    outputStream.putFrame(img)