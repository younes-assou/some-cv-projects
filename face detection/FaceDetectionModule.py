import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetection=0.8):
        self.minDetection=minDetection

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetector = self.mpFaceDetection.FaceDetection(minDetection)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceDetector.process(frameRGB)
        bboxs =[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                fh, fw, fc = frame.shape
                bbox = int(bboxC.xmin*fw), int(bboxC.ymin*fh), int(bboxC.width*fw), int(bboxC.height*fh)
                if draw:
                    frame = self.fancyDraw(frame, bbox)
                    cv.putText(frame, f'{int(detection.score[0]*100)}', (bbox[0],bbox[1]), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)
                bboxs.append([bbox, detection.score])
        
        return frame, bboxs

    def fancyDraw(self, frame, bbox, l=30, th=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        
        cv.rectangle(frame, bbox, (0,255,0), rt)
        #top lef
        cv.line(frame, (x,y), (x+l,y), (0,255,0), th)
        cv.line(frame, (x,y), (x,y+l), (0,255,0), th)
        #top right
        cv.line(frame, (x1-l,y), (x1,y), (0,255,0), th)
        cv.line(frame, (x1,y), (x1,y+l), (0,255,0), th)
        #bottom lef
        cv.line(frame, (x,y1), (x+l,y1), (0,255,0), th)
        cv.line(frame, (x,y1), (x,y1-l), (0,255,0), th)
        #bottom right
        cv.line(frame, (x1-l,y1), (x1,y1), (0,255,0), th)
        cv.line(frame, (x1,y1), (x1,y1-l), (0,255,0), th)
        

        return frame


def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()

    while True:
        succes, frame = cap.read()
        frame, bboxs = detector.findFaces(frame)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cv.imshow('Video', frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()