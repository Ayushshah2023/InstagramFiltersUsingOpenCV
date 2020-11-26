import cv2
import sys
img=cv2.imread("E:\DIP\Assign_01\q1.jpg")
S = 255
img = cv2.imread(img,cv2.IMREAD_GRAYSCALE) 
negative_img = S - img
cv2.imwrite("negative_"+sys.argv[1], negative_img)
threshold_img = (img > 127) * S
cv2.imwrite("threshold_"+sys.argv[1], threshold_img)
