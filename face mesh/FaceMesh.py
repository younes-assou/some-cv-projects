import cv2 as cv
import mediapipe as mp
import time

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
mpDrawSpec = mpDraw.DrawingSpec(1,1)


cap = cv.VideoCapture(0)

pTime = 0

while True:
    succes, frame = cap.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, mpDrawSpec, mpDrawSpec)
            

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)


    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break


cap.release()
cv.destroyAllWindows()