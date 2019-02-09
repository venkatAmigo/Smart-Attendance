# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:56:55 2018

@author: venkat
"""


import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("test2.mp4")
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("trainingdata.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==2):
            id="alok"
        if id==1:
            id="alok"
        if id==3:
            id="anjali"
        if id==4:
            id="Gaurav"
        if id==5:
            id='rahul'
        if id==6:
            id="akshay"
        cv2.putText(img,str(id),(x,y+h),font,2,cv2.COLOR_BGR2HSV,2)
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()