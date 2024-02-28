import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
imgBG=cv2.imread("deneme.jpg")
segmentor=SelfiSegmentation()



while True:
    success, img=cap.read()
    imgout=segmentor.removeBG(img,imgBG,cutThreshold=0.8)

    cvzone.stackImages([img,imgout],2,1)
    cv2.imshow("Image",img)
    cv2.imshow("Image Out",imgout)
    cv2.waitKey(1)