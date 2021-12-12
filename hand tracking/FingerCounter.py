import cv2 as cv
import time
import math
import numpy as np
import HandTrackingModule as htm

#############################
wCam, hCam = 1040, 680
#############################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detectionConf=0.7)
tipIds = [4, 8, 12, 16, 20]

while True:
    succes, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    fingers =[]
    if len(lmList)!=0:

        #thumb
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        #4 fingers
        for id in tipIds[1:]:
            if lmList[id][2] < lmList[id-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    frame = cv.flip(frame, 1)
    cv.putText(frame, str(int(fingers.count(1))), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
    cv.imshow('Video', frame)

    cv.waitKey(1)
