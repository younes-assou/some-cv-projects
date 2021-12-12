import cv2 as cv
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackingConf=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConf=detectionConf
        self.trackingConf=trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        
        if self.results.multi_hand_landmarks:
            for handLdm in self.results.multi_hand_landmarks:
                if draw: 
                    self.mpDraw.draw_landmarks(frame, handLdm, self.mpHands.HAND_CONNECTIONS)

        return frame
    
    def findPosition(self, frame, handNo=0, draw=True):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx,cy), 5, (255,0,255), 10, cv.FILLED)

        return lmList

def main():
    capture = cv.VideoCapture(0)
    detector = HandDetector()
    pTime = 0

    while True :

        isTrue, frame = capture.read()
        detector.findHands(frame, draw=False)
        lmList = detector.findPosition(frame)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cv.imshow('Video', frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()