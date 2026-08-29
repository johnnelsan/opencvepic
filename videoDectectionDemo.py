import cv2 
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
faceModel = YOLO('yolov8m-face.pt')





capture = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord("x"):
   _, frame = capture.read()

   result = model(frame)
   faceResult = faceModel(frame)

   personBody = result[0].plot()
   # Using the plot from the person body to also show how person face is working
   personFace = faceResult[0].plot(img=personBody)
   cv2.imshow('my window', personFace)
   cv2.moveWindow('my window', 0, 0)

cv2.waitKey(5000)
capture.release()
cv2.destroyAllWindows()


