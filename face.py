#Face detection program which detect that in front of webcam there is any human face or not? and if than it highlights it.

#First of all we have to import all required libraries
import cv2       
import numpy as np


#Using haarcascade file  we have to create two classifier object one for face and another one is for eyes.
faceclassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeclassifier = cv2.CascadeClassifier("haarcascade_eye.xml")


#To turn on the camera, call the videocapture function and give it the Camera id as a parameter.
cap = cv2.VideoCapture(0)


while True:
    rat,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceclassifier.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roigray = gray[y:y+h,x:x+w]
        roicolor = img[y:y+h,x:x+w]
        eyes = eyeclassifier.detectMultiScale(roigray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roicolor,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)   #Displays an image in the specified window. 
    k = cv2.waitKey(1)  #The function waitKey waits for a key event infinitely
    if k == 13 or 27:     # Press enter(13) or Esc(27) for exit. 
        break


#For release the camera 
cap.release()

#To close all windows
cv2.destroyAllWindows()
