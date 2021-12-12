import cv2 as cv
import mediapipe as mp
import time


class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionConf=0.5, trackingConf=0.5):

        self.mode=mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionConf=detectionConf
        self.trackingConf=trackingConf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return frame


    def findPosition(self, frame, draw=False):
        lmList =[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx,cy), 5, (255,0,255), 10, cv.FILLED)
        return lmList


def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = PoseDetector()

    while True:
        succes, frame = cap.read()
        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cv.imshow('Video', frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()