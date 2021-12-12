import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


cap = cv.VideoCapture(0)

pTime = 0

while True:
    succes, frame = cap.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            #cv.circle(frame, (cx,cy), 5, (255,0,255), 10, cv.FILLED)




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)


    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break


cap.release()
cv.destroyAllWindows()