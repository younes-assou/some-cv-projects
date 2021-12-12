import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 4, 0.5, 0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True :
    isTrue, frame = capture.read()

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    
    if results.multi_hand_landmarks:
        for handLdm in results.multi_hand_landmarks:
            for id, lm in enumerate(handLdm.landmark):
                #print(id, lm)
                h,w,c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                
                cv.circle(frame, (cx,cy), 5, (255,0,255), 10, cv.FILLED)
            mpDraw.draw_landmarks(frame, handLdm, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)


    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()