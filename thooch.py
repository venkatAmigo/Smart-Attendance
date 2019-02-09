# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:26:51 2018

@author: venkat
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 18:44:06 2018

@author: venkat
"""

import os

import numpy as np

import cv2

from PIL import Image # For face recognition we will the the LBPH Face Recognizer 

recognizer = cv2.face.LBPHFaceRecognizer_create();

path="I://"

def getImagesWithID(path):

    #imagePaths = [os.path.join(path, f) for f in os.listdir(path)]   
    imagePath="I:/BTECH ADS/project/User.1.1.jpg"
 # print image_path   

 #getImagesWithID(path)

    faces = []

    IDs = []

  # Read the image and convert to grayscale

    facesImg = Image.open("I:/BTECH ADS/project/User.1.1.jpg").convert('L')

    faceNP = np.array(facesImg, 'uint8')

        # Get the label of the image

    ID= int(os.path.split(imagePath)[-1].split(".")[1])

         # Detect the face in the image

    faces.append(faceNP)

    IDs.append(ID)

    cv2.imshow("Adding faces for traning",faceNP)

    cv2.waitKey(10)

    return np.array(IDs), faces

Ids,faces  = getImagesWithID(path)

recognizer.train(faces,Ids)

recognizer.save("trainingdata.yml")

cv2.destroyAllWindows()