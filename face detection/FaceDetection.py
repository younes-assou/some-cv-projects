import cv2 as cv
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
faceDetector = mpFaceDetection.FaceDetection(0.8)
mpDraw = mp.solutions.drawing_utils


cap = cv.VideoCapture(0)

pTime = 0

while True:
    succes, frame = cap.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceDetector.process(frameRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            fh, fw, fc = frame.shape
            bbox = int(bboxC.xmin*fw), int(bboxC.ymin*fh), int(bboxC.width*fw), int(bboxC.height*fh)
            cv.rectangle(frame, bbox, (0,255,0), 2)
            cv.putText(frame, f'{int(detection.score[0]*100)}', (bbox[0],bbox[1]), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)


    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break


cap.release()
cv.destroyAllWindows()