import cv2 as cv
import numpy as np
import hand as ht
from math import sqrt
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import autopy

#------------Volume--------------
# PyCaw Initializations
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Getting Volume Range
volRange = volume.GetVolumeRange() 
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0


def volCondition(lmList):
    # Making a Condition that only when index and thumb are open, volcontrol will commence
    fingerCheck = detect.fingerCheck(lmList)  
    
    if fingerCheck[0] and not(fingerCheck[2] and fingerCheck[3] and fingerCheck[4]):
        volControl = True
    else: 
        volControl = False
    
    # Pinky Check
    if fingerCheck[4] and volControl:
        pinky = True
    else:
        pinky = False

    return volControl, pinky


def calcDistance(frame, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Calculating Midpoint
    mx, my = (x1+x2)/2, (y1+y2)/2 

    # For Min (0%) Volume
    if distance < 30:
        cv.circle(frame, (int(mx), int(my)), 7, (0, 0, 255), -1)

    # For Max (100%) Volume
    elif distance > 180:
        cv.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv.circle(frame, (int(mx), int(my)), 7, (255, 0, 0), -1)
        cv.circle(frame, (x1, y1), 7, (255, 0, 0), -1)
        cv.circle(frame, (x2, y2), 7, (255, 0, 0), -1)
    
    # For inBetween Values
    else:
        cv.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv.circle(frame, (int(mx), int(my)), 7, (255, 255, 0), -1)
        cv.circle(frame, (x1, y1), 7, (255, 255, 0), -1)
        cv.circle(frame, (x2, y2), 7, (255, 255, 0), -1)

    return distance


def volGraphics(frame, distance):
    # Mapping the Values like in Arduino
    volPer = np.interp(distance, [30, 180], [0, 100])
    volBar = np.interp(volPer, [0, 100], [400, 150])
    
    # Setting the volume at every 5 Levels 
    smoothness = 5
    volPer = smoothness * round(volPer/smoothness)
    
    if pinky:
        # Setting the Desired Volume
        volume.SetMasterVolumeLevelScalar(volPer/100, None)
    
    getVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv.putText(frame, f"Volume Set {getVol}%", (320, 20), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)

    # Drawing the Volume Bar 
    cv.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
    cv.rectangle(frame, (50, int(volBar)), (85, 400), (255, 0, 0), -1)
    cv.putText(frame, f"{int(volPer)}%", (40, 450), cv.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    return frame

#--------------------Volume--------------


#-------------------Hand Detection------------------------
wCam, hCam = 640, 480
frameR = 100     #Frame Reduction
smoothening = 7  #random value

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detect = ht.HandDetector(detectConfi=0.7, maxHands=1)
wScr, hScr = autopy.screen.size()

#---------------------Hand Detection-------------



#-------------------Combining---------------
while cap.isOpened():
    isSuccess, frame = cap.read()

    if isSuccess:
        # Fliping the frame horrizontally
        frame = cv.flip(frame, 1)

        #Step 1 FInd the Land Marks
        frame = detect.findHands(frame)
        lmList, bBox = detect.findPosition(frame)

        if len(lmList) != 0:
            
            # Condition Check for Volume
            volControl, pinky = volCondition(lmList)

            # condition mouse control
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            fingers = detect.fingersUp()
            cv.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

            #Volume ifelse cond 
            if volControl:
                tx, ty = lmList[4][1], lmList[4][2]
                ix, iy = lmList[8][1], lmList[8][2]

                distance = calcDistance(frame, [tx, ty], [ix, iy])
                frame = volGraphics(frame, distance)
                
                cv.putText(frame, f"Volume Set ON", (bBox[0]-30, bBox[1]-30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            else:
                cv.putText(frame, f"Volume Set OFF", (bBox[0]-30, bBox[1]-30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            #mouse ifelse cond
            # Step4: Only Index Finger: Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:

                # Step5: Convert the coordinates
                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

                # Step6: Smooth Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Step7: Move Mouse
                autopy.mouse.move(wScr - clocX, clocY)
                cv.circle(frame, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                plocX, plocY = clocX, clocY

            # Step8: Both Index and middle are up: Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:

                # Step9: Find distance between fingers
                length, frame, lineInfo = detect.findDistance(8, 12, frame)

                # Step10: Click mouse if distance short
                if length < 40:
                    cv.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
                    autopy.mouse.click()
        
        # Showing the FPS
        detect.addFPS(frame)

        # Showing the Frames
        cv.imshow("Video", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break
            


cap.release()
cv.destroyAllWindows()