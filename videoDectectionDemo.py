import cv2 
from ultralytics import YOLO
# importing the YOLO models  
model = YOLO('yolov8n.pt')
faceModel = YOLO('yolov8m-face.pt')

## capturing the video feed from the webcam
capture = cv2.VideoCapture(0)

#while the user doesn't press x 
while cv2.waitKey(1) != ord("x"):
   # get each frame from the capture.read method
   _, frame = capture.read()
   #first model the body/item of the things 
   result = model(frame)
   #then model the faces of dectected persons
   faceResult = faceModel(frame)
   #plot the recognized items
   personBody = result[0].plot()
   # Using the plot from the person body to also show how person face is working
   personFace = faceResult[0].plot(img=personBody)
   # draw the results 
   cv2.imshow('SAY HELLO!', personFace)
   cv2.moveWindow('SAY HELLO!', 0, 0)

cv2.waitKey(1000)
capture.release()
cv2.destroyAllWindows()


