import cv2 
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
faceModel = YOLO('yolov8m-face.pt')

photo = cv2.imread('photo1.jpg')
result = model(photo)
faceResult = faceModel(photo)

personBody = result[0].plot()
# Using the plot from the person body to also show how person face is working
personFace = faceResult[0].plot(img=personBody)

cv2.imshow('my window', personFace)
cv2.moveWindow('my window', 1800, 200)

cv2.waitKey(5000)
cv2.destroyAllWindows()
