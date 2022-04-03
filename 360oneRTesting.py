
import cv2
import numpy

scale_factor = 1

vid = cv2.VideoCapture(0)
ret, test_frame = vid.read()

x_res = int(test_frame.shape[1]/scale_factor)
y_res = int(test_frame.shape[0]/scale_factor)
print(x_res,y_res)

while(True):

    ret, frame = vid.read()
    
    cv2.imshow("res",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

vid.release()
cv2.destroyAllWindows()