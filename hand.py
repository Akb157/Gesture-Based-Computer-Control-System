import cv2 as cv
import mediapipe as mp
from matplotlib.transforms import Bbox as bBox
import time as t
import math
from math import sqrt
import numpy as np

##volume

class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectConfi=0.5, trackConfi=0.5):
        # Initializig the Modes
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectConfi = detectConfi
        self.trackConfi = trackConfi
        
        # Initializing the Mediapipe
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectConfi, self.trackConfi)
        self.tipIds = [4, 8, 12, 16, 20]
        # Initializing FPS Counter
        self.pTime = 0
        self.cTime = 0



    def findHands(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Getting the Landmarks
        self.results = self.hands.process(frameRGB)
        # print(results.multi_hand_landmarks)

        # Drawing the Landmarks on Hands
        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLMs, self.mpHands.HAND_CONNECTIONS)
        return frame
    

    def findPosition(self, frame, handNo=0, boxDraw=False):
        lmList = []
        bBox = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # Giving Every point(landmarks) on the hand their IDs
            for id, lm in enumerate(myHand.landmark):
                # As the x and y values are in aspect ratio we convert them to pixel    
                h, w, _ = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            
            # Min and Maximum for cx and cy values
            _, xMin, yMin = np.min(lmList, axis=0)
            _, xMax, yMax = np.max(lmList, axis=0)

            bBox = [xMin, yMin, xMax, yMax]

            if boxDraw:
                cv.rectangle(frame, (bBox[0]-30, bBox[1]-30), (bBox[2]+30, bBox[3]+30), (0, 255, 0), 3)
        if boxDraw:
            return lmList, bBox
        else:
            return lmList


    def fingerCheck(self, lmList):
        tipIDs = [4, 8, 12, 16, 20]
        fCheck = []

        # For thumb
        if lmList[4][1] < lmList[3][1]:
            fCheck.append(True)
        else:
            fCheck.append(False)

        # For other fngers
        for id in range(1, 5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fCheck.append(True)
            else:
                fCheck.append(False)

        return fCheck


    def calcArea(self, bBox):
        area = 0
        if len(bBox) == 4:
            xMin, xMax = bBox[0], bBox[2]
            yMin, yMax = bBox[1], bBox[3]

            boxHeight = yMax - yMin
            boxWidth = xMax - xMin

            area = boxWidth * boxHeight
        return area


    def calcDistance(self, frame, pt1, pt2, draw=False):
        x1, y1 = pt1
        x2, y2 = pt2
        distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Calculating Midpoint
        mx, my = (x1+x2)/2, (y1+y2)/2 

        if draw:
            cv.line(frame, pt1, pt2, (255, 255, 0), 3)

        return distance

    
    def addFPS(self, frame):
        # Calculating the FPS
        self.cTime = t.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime * 0.90 + self.pTime * 0.10
        cv.putText(frame, f"FPS: {int(fps)}", (10, 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)



        

    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bBox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bBox = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bBox

    def fingersUp(self):
        fingers = []
    # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
    # totalFingers = fingers.count(1)
        return fingers

    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(frame, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(frame ,(x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]



def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detect = HandDetector()

    pTime = 0
    cTime = 0

    while(cap.isOpened()):
        isSuccess, frame = cap.read()

        if isSuccess:
            # Fliping the frame horrizontally
            frame = cv.flip(frame, 1)
            frame = detect.findHands(frame)
            lmList = detect.findPosition(frame, boxDraw=False)

            if len(lmList) != 0:
                print(lmList[4])

        detect.addFPS(frame)    
        cTime = t.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)



        cv.imshow("Video", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()























        
