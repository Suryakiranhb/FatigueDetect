from flask import Flask, render_template, Response, request
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import sys

app = Flask(__name__)

def turnOff():
    sys.exit()

def sound_alarm(path):
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	a = dist.euclidean(eye[1], eye[5])
	b = dist.euclidean(eye[2], eye[4])

	c = dist.euclidean(eye[0], eye[3])

	ear = (a + b) / (2.0 * c)
	return ear

def mouthAspectRatio(mouth):
	a = dist.euclidean(mouth[1], mouth[7])
	b = dist.euclidean(mouth[2], mouth[6])
	c = dist.euclidean(mouth[3], mouth[5])
	d = dist.euclidean(mouth[0], mouth[4])

	mar = (a + b + c) / (2.0 * d)
	return mar

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
earThreshold = 0.3
earConsecutiveFrames = 30

marThreshold = 0.3
marConsecutiveFrames = 30

eyeCounter = 0
mouthCounter = 0
alarm = False

print("Process : loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(innerMouthStart, innerMouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

print("Process : starting video stream thread...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)

def start():
    while True:
        
        ok, frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[leftEyeStart:leftEyeEnd]
            rightEye = shape[rightEyeStart:rightEyeEnd]
            innerMouth = shape[innerMouthStart:innerMouthEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mar = mouthAspectRatio(innerMouth)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(innerMouth)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            if ear < earThreshold or mar > marThreshold:
                eyeCounter += 1
                mouthCounter += 1

                if eyeCounter >= earConsecutiveFrames and mouthCounter >= marConsecutiveFrames:
                    if not alarm:
                        alarm = True

                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm,
                                args=(args["alarm"],))
                            t.deamon = True
                            t.start()

                    cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                eyeCounter = 0
                mouthCounter = 0
                alarm = False

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        returnValue, buffer = cv2.imencode('.jpeg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output')
def output():
    return Response(start(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/off')
def off():
    return Response(turnOff(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug = True)