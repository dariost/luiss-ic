#!/usr/bin/env python3

# License: Apache 2.0
# Copyright 2018 - Dario Ostuni <dario.ostuni@gmail.com>
# Copyright 2018 - Edoardo Morassutto <edoardo.morassutto@gmail.com>

import cv2
import sys
import numpy as np

THRESHOLD = 15
BLUR_LEVEL = 15
DILATE_ITERATIONS = 5
HARDNESS = 2

def checkExit():
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        sys.exit(0)

def getFrame(camera):
    grabbed, frame = camera.read()
    if not grabbed:
        print("WTF²?")
        sys.exit(1)
    return frame

def processFrame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if BLUR_LEVEL > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_LEVEL, BLUR_LEVEL), 0)
    return gray

if __name__ == "__main__":
    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    camera = cv2.VideoCapture(0)
    prevFrame = processFrame(getFrame(camera))
    prevPoints = None
    bitmask = cv2.absdiff(prevFrame, prevFrame)
    while True:
        rawFrame = getFrame(camera)
        frame = processFrame(rawFrame)
        diff = cv2.absdiff(prevFrame, frame)
        thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
        bitmask = np.maximum(bitmask, thresh)
        softBitmask = bitmask // HARDNESS
        rawFrame[..., 0] += softBitmask
        rawFrame[rawFrame[..., 0] < softBitmask, 0] = 255
        cv2.imshow("Tracker", rawFrame)
        checkExit()
        prevFrame = frame
        points = np.count_nonzero(bitmask)
        if prevPoints != points:
            prevPoints = points
            print("Points:", points)
