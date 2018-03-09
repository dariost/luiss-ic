#!/usr/bin/env python3

# License: Apache 2.0
# Copyright 2018 - Dario Ostuni <dario.ostuni@gmail.com>
# Copyright 2018 - Edoardo Morassutto <edoardo.morassutto@gmail.com>

import cv2
import sys
import numpy as np
import time
import subprocess
import platform

WIDTH = 800
HEIGHT = 600
CAMERA = 1
SCORE_MULT = 1.0
DURATION = 120
START_ON_MOVE = True

THRESHOLD = 15
BLUR_LEVEL = 15
DILATE_ITERATIONS = 5
HARDNESS = 2

COLOR_R = 220 / 255
COLOR_G =  16 / 255
COLOR_B = 193 / 255

started = False
startTime = None
reset = False

def processInput():
    global started
    global startTime
    global reset
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        sys.exit(0)
    if key == ord(" ") and not started:
        started = True
        startTime = time.time()
        print("Started")
    elif key == ord(" ") and started:
        startTime -= DURATION
    if key == ord("r"):
        reset = True
    try:
        if cv2.getWindowProperty("Tracker", 0) < 0:
            sys.exit(0)
    except:
        sys.exit(0)

def getFrame(camera):
    grabbed, frame = camera.read()
    if not grabbed:
        print("WTF²?")
        sys.exit(1)
    return frame

def preprocessFrame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if BLUR_LEVEL > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_LEVEL, BLUR_LEVEL), 0)
    return gray

def processFrame(rawFrame, prevFrame, bitmask):
    frame = preprocessFrame(rawFrame)
    diff = cv2.absdiff(prevFrame, frame)
    thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
    bitmask = np.maximum(bitmask, thresh)
    return frame, bitmask

def scaleTuple(f, t):
    return (int(t[0] * f), int(t[1] * f), int(t[2] * f))

def sumTuple(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2])

def convertColor(perc):
    if perc < 0.5:
        return sumTuple(scaleTuple(2 * perc, (0, 255, 255)), scaleTuple(1 - 2 * perc, (0, 0, 255)))
    else:
        return sumTuple(scaleTuple(2 * (perc - 0.5), (0, 255, 0)), scaleTuple(1 - 2 * (perc - 0.5), (0, 255, 255)))

if __name__ == "__main__":
    if platform.system() in ("Linux",):
        ffout = subprocess.run(["ffmpeg",  "-f",  "v4l2", "-list_formats", "all", "-i", "/dev/video" + str(CAMERA)], stderr=subprocess.PIPE)
        resolutions = [tuple(map(int, y)) for y in [x.split("x") for x in ffout.stderr.decode().split("\n")[-3].split(":")[-1].strip().split(" ")]]
        if (WIDTH, HEIGHT) not in resolutions:
            print((WIDTH, HEIGHT), "resolution is not supported")
            print("Supported resolution are:", resolutions)
            sys.exit(1)
    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    camera = cv2.VideoCapture(CAMERA)
    camera.set(3, WIDTH)
    camera.set(4, HEIGHT)
    prevPoints = None
    stopped = False
    while True:
        processInput()
        rawFrame = getFrame(camera)
        if reset:
            reset = False
            prevFrame = preprocessFrame(getFrame(camera))
            bitmask = cv2.absdiff(prevFrame, prevFrame)
            startTime = None
            prevPoints = None
            stopped = False
            started = False
            print("Reset")
        elif not started:
            prevFrame = preprocessFrame(getFrame(camera))
            bitmask = cv2.absdiff(prevFrame, prevFrame)
        elif startTime + DURATION > time.time():
            frame, bitmask = processFrame(rawFrame, prevFrame, bitmask)
            prevFrame = frame
        elif not stopped:
            print("Stopped")
            stopped = True

        softBitmask = bitmask // HARDNESS
        rawFrame[..., 0] += (softBitmask * COLOR_B).astype(np.uint8)
        rawFrame[rawFrame[..., 0] < (softBitmask * COLOR_B).astype(np.uint8), 0] = 255
        rawFrame[..., 1] += (softBitmask * COLOR_G).astype(np.uint8)
        rawFrame[rawFrame[..., 1] < (softBitmask * COLOR_G).astype(np.uint8), 1] = 255
        rawFrame[..., 2] += (softBitmask * COLOR_R).astype(np.uint8)
        rawFrame[rawFrame[..., 2] < (softBitmask * COLOR_R).astype(np.uint8), 2] = 255
        points = np.count_nonzero(bitmask)

        if startTime is not None:
            if START_ON_MOVE and points == 0:
                startTime = time.time()
            perc = max(0, startTime + DURATION - time.time()) / DURATION
            cv2.putText(rawFrame, "Score: %d (%.2f%%)" % (int(SCORE_MULT * points), points * 100 / WIDTH / HEIGHT), (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, convertColor(points / WIDTH / HEIGHT))
            cv2.putText(rawFrame, "Time: %.2fs" % (perc * DURATION), (20, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, convertColor(perc))
        cv2.imshow("Tracker", rawFrame)
        if prevPoints != points:
            prevPoints = points
            print("Points:", int(SCORE_MULT * points))
