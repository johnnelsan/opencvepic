import cv2 

photo = cv2.imread('photo1.jpg')
cv2.imshow('my window', photo)
cv2.moveWindow('my window', 100, 100)

cv2.waitKey(5000)
cv2.destroyAllWindows()