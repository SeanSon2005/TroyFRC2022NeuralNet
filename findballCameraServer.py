import cv2
import numpy as np
import time
from cscore import CameraServer
from networktables import NetworkTable, NetworkTables
import threading

blue_ball = True #determine ally color
testing_on_computer = True #testing on roborio or computer

# distance from camera to object(face) measured
# centimeter
Known_distance = 40
 
# width of face in the real world or Object Plane
# centimeter
Known_width = 24.13
 
# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
 
# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX
 
# face detector object
face_detector = cv2.CascadeClassifier("girth.xml")

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

#cs = CameraServer.getInstance()
#cs.enableLogging()

#camera = cs.startAutomaticCapture()

#cvSink = cs.getVideo()
img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

#ret, test_frame = cvSink.grabFrame(img)

#x_res = int(test_frame.shape[1]/scale_factor)
#y_res = int(test_frame.shape[0]/scale_factor)

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

# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
 
    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length
 
# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
 
    distance = (real_face_width * Focal_Length)/face_width_in_frame
 
    # return the distance
    return distance
 
 
def face_data(image):
 
    face_width = 0  # making face width to zero
 
    # converting color image ot gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
 
    # looping through the faces detect in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:
 
        # draw the rectangle on the face
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
 
        # getting face width in the pixels
        face_width = w
 
    # return the face width in pixel
    return face_width

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

# reading reference_image from directory
ref_image = cv2.imread("ball.jpg")


 
# find the face width(pixels) in the reference_image
ref_image_face_width = face_data(ref_image)
 
# get the focal by calling "Focal_Length_Finder"
# face width in reference(pixels),
# Known_distance(centimeters),
# known_width(centimeters)
Focal_length_found = Focal_Length_Finder(
    Known_distance, Known_width, ref_image_face_width)
 
print(Focal_length_found)
 
# show the reference image
cv2.imshow("ball.jpg", ref_image)
 
# initialize the camera object so that we
# can get frame from it
cap = cv2.VideoCapture(0)
 
# looping through frame, incoming from
# camera/video
def calculateDistance():
    Distance = None
    # reading the frame from camera
    _, frame = cap.read()
 
    # calling face_data function to find
    # the width of face(pixels) in the frame
    face_width_in_frame = face_data(frame)
 
    # check if the face is zero then not
    # find the distance
    if face_width_in_frame != 0:
       
        # finding the distance by calling function
        # Distance distance finder function need
        # these arguments the Focal_Length,
        # Known_width(centimeters),
        # and Known_distance(centimeters)
        Distance = Distance_finder(
            Focal_length_found, Known_width, face_width_in_frame)
 
        # draw line as background of text
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
 
        # Drawing Text on the screen
        cv2.putText(
            frame, f"Distance: {round(Distance,2)} CM", (30, 35),
          fonts, 0.6, GREEN, 2)
 
    # show the frame on the screen
    cv2.imshow("frame", frame)
    return Distance
    # quit the program if you press 'q' on keyboard
    # if cv2.waitKey(1) == ord("q"):
        
while True:
  calculateDistance()
'''      
if(blue_ball):
  while True:
    #getting distance
    distance = calculateDistance()
    print(distance)

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
        adjustment = PIDCalc(i[0]) / distance**2
        vision_nt.putNumber('PID', adjustment)

else:
  while True:
    #getting distance
    distance = calculateDistance()

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
'''