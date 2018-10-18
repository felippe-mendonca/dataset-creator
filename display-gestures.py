import os
import cv2
import json
import time
import numpy as np
from sys import exit
from video_loader import VideoLoader
from is_wire.core import Logger

log = Logger(name="DisplayGestures")

with open('gestures.json', 'r') as f:
    gestures = json.load(f)
keys = list(map(lambda x: '{:x}'.format(int(x)), gestures.keys()))

gesture_id = 0
video_file = 'samples/{:02d}.MOV'
vl = VideoLoader(video_file.format(1))
default_screen = np.zeros(vl.resolution()).transpose()

t0 = 0
while True:
    if gesture_id > 0:
        image = next(vl)
        image[:50, :, :] = 0
        image[-50:, :, :] = 0
        text = "GESTURE_ID: {:02d} ({:s})".format(gesture_id,
                                                  gestures[str(gesture_id)])
        fontFace = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img=image, text=text, org=(50, 35), fontFace=fontFace, fontScale=1.0, color=(255,255,255))
        cv2.imshow('', image)
    else:
        cv2.imshow('', default_screen)

    key = cv2.waitKey(1)

    t1 = time.time()
    time.sleep(max(0.0, 1.0 / vl.fps() - 1e-3 - (t1 - t0)))
    t0 = time.time()

    if key == -1:
        continue
    if chr(key) in keys:
        gesture_id = int(chr(key), 16)
        vl.load(video_file.format(gesture_id))
    if key == ord('0'):
        gesture_id = 0
    if key == ord('q'):
        exit(0)
