import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, mode=False, maxFaces=2, detectionConf=0.5, trackingConf=0.5):

        self.mode=mode
        self.maxFaces=maxFaces
        self.detectionConf=detectionConf
        self.trackingConf=trackingConf

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawSpec = self.mpDraw.DrawingSpec(1,1)


    def findFaceMesh(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(frameRGB)
        faces =[]
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.mpDrawSpec, self.mpDrawSpec)
                
                face =[]
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = frame.shape
                    x, y = int(lm.x*w), int(lm.y*h)
                    face.append([x, y])
                faces.append(face)
        
        return frame, faces
            



def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        succes, frame = cap.read()
        frame, faces = detector.findFaceMesh(frame)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cv.imshow('Video', frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()