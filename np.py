# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import math
import argparse
import datetime
import imutils
import time
import cv2
from pyfirmata import Arduino, util
from time import sleep
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BOARD)

GPIO.setup(11,GPIO.OUT)
servo1 = GPIO.PWM(11,50) # pin 11 for servo1
GPIO.setup(12,GPIO.OUT)
servo2 = GPIO.PWM(12,50) # pin 12 for servo2

servo1.start(0)
servo2.start(0)

stdDev = [10, 20]

def xy_2_py(xNext, yNext): # Convert x-y to pitch-yaw
    fovX = 62.2      # Horizontal FOV of Camera
    # 53.5
    fovY = 48.8      # Vertical FOV of Camera
    #41.41
    maxHPixel = 720            # Horizontal Number of Pixels
    maxVPixel = 1280           # Vertical Number of Pixels
    
    # Pixel Ratios
    horiDegPixel = fovX / maxHPixel
    vertDegPixel = fovY / maxVPixel
    
    # Angles
    yaw = (maxHPixel - xNext)*horiDegPixel
    pitch = (maxVPixel - yNext)*vertDegPixel
    return [yaw, pitch]


def nextPos(boxPos, stdDev):
    return np.random.normal(boxPos, stdDev)


def increase_brightness(frame, value=60):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return frame 


#def output(x,y):

#    return

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied text
    frame = vs.read()
    frame = increase_brightness(frame, value=60)
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"
    
    
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Find the largest box
    lgst_area = 0.0;
    lgst_param = [0.0, 0.0, 0.0, 0.0]; #x, y, w, h
    for box in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(box) < args["min_area"]:
            continue
        
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(box)
  

        if (lgst_area < w*h):
            lgst_area = w*h
            lgst_param = [x, y, w, h]
    
    x = int(lgst_param[0]);
    y =int(lgst_param[1]);
    w =int(lgst_param[2]);
    h = int(lgst_param[3]);
    
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)


    xMid0 = x + w/2
    yMid0 = y + h/2
    
    xMid = xMid0/10+2.5
    yMid = yMid0/10+2.5

    #print([xMid, yMid])
    v = nextPos([xMid, yMid], stdDev)
    angles = xy_2_py(v[0], v[1])
    if xMid < 4:
        servo2.ChangeDutyCycle(xMid)
        print("x")
        print(xMid)
    else:
        servo2.ChangeDutyCycle(4)

    if yMid < 8:
        servo1.ChangeDutyCycle(yMid)
        print("y")
        print(yMid)
    else:
        servo1.ChangeDutyCycle(8)
    time.sleep(0.8)
    cv2.imshow("cat feed", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()



