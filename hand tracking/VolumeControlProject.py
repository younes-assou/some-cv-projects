import cv2 as cv
import time
import math
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################
wCam, hCam = 1040, 680
#############################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detectionConf=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMasterVolumeLevel()
minVol, maxVol = volume.GetVolumeRange()[:2]

volBar = 400
volPer = 0

while True:
    succes, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList)!=0 :
        #print(lmList[4],lmList[8])
        x1,y1 = lmList[4][1:]
        x2,y2 = lmList[8][1:]
        cx,cy = (x1+x2)//2, (y1+y2)//2
        cv.circle(frame, (x1,y1), 10, (0,255,0), cv.FILLED)
        cv.circle(frame, (x2,y2), 10, (0,255,0), cv.FILLED)
        cv.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv.circle(frame, (cx,cy), 10, (0,255,0), cv.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        # hand range 50 -- 300
        # volume range -65 ---- 0
        vol = np.interp(length, [50,300],[minVol, maxVol])
        volBar = np.interp(length, [50,300],[400, 150])
        volPer = np.interp(length, [50,300],[0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv.circle(frame, (cx,cy), 10, (0,0,255), cv.FILLED)

    cv.rectangle(frame, (50,150), (85,400),(0,255,0), 2)
    cv.rectangle(frame, (50,int(volBar)), (85,400),(0,255,0), cv.FILLED)
    cv.putText(frame, str(int(volPer))+'%', (55,420), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    frame = cv.flip(frame, 1)
    cv.imshow('Video', frame)

    cv.waitKey(1)
