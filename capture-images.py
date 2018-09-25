import re
import os
import sys
import shutil
from subprocess import Popen, PIPE, STDOUT
import time
import cv2
import numpy as np
from is_msgs.image_pb2 import Image
from is_wire.core import Channel, Subscription, Message, Logger
from options_pb2 import DatasetCaptureOptions
from google.protobuf.json_format import Parse


def get_id(topic):
    match_id = re.compile(r'CameraGateway.(\d+).Frame')
    match = match_id.search(msg.topic)
    if not match:
        return None
    else:
        return int(match.group(1))


def place_images(output_image, images):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0:h, 0:w, :] = images[0]
    output_image[0:h, w:2 * w, :] = images[1]
    output_image[h:2 * h, 0:w, :] = images[2]
    output_image[h:2 * h, w:2 * w, :] = images[3]


log = Logger(name='Capture')

if len(sys.argv) != 3:
    log.critical("Usage: python3 capture.py <PERSON_ID> <GESTURE_ID>")
    sys.exit(-1)

person_id = int(sys.argv[1])
gesture_id = int(sys.argv[2])
log.info("PERSON_ID: {} GESTURE_ID: {}", person_id, gesture_id)

with open('options.json', 'r') as f:
    try:
        options = Parse(f.read(), DatasetCaptureOptions())
    except Exception as ex:
        log.critical('Unable to read \"options.json\". \n{}', ex)
        sys.exit(-1)

if not os.path.exists(options.folder):
    os.makedirs(options.folder)

sequence_folder = os.path.join(options.folder, 'p{:03d}g{:02d}'.format(
    person_id, gesture_id))

if os.path.exists(sequence_folder):
    log.warn(
        'Path to PERSON_ID={} GESTURE_ID={} already exists.\nWould you like to proceed? All data will be deleted! [y/n]',
        person_id, gesture_id)
    key = input()
    if key == 'y':
        shutil.rmtree(sequence_folder)
    elif key == 'n':
        sys.exit(0)
    else:
        log.critical('Invalid command \'{}\', exiting.', key)
        sys.exit(-1)

os.makedirs(sequence_folder)

channel = Channel(options.broker_uri)
subscription = Subscription(channel)
for camera in options.cameras:
    subscription.subscribe('CameraGateway.{}.Frame'.format(camera.id))

size = (2 * options.cameras[0].config.image.resolution.height,
        2 * options.cameras[0].config.image.resolution.width, 3)
full_image = np.zeros(size, dtype=np.uint8)

images_data = {}
images = {}
n_sample = 0
display_rate = 2
while True:
    msg = channel.consume()
    camera = get_id(msg.topic)
    if camera is None:
        continue
    pb_image = msg.unpack(Image)
    if pb_image is None:
        continue
    data = np.fromstring(pb_image.data, dtype=np.uint8)
    images_data[camera] = data

    if len(images_data) == len(options.cameras):
        # save images
        for camera in options.cameras:
            filename = os.path.join(
                sequence_folder, 'c{:02d}s{:08d}.jpeg'.format(
                    camera.id, n_sample))
            with open(filename, 'wb') as f:
                f.write(images_data[camera.id])
        # display images
        if n_sample % display_rate == 0:
            images = [cv2.imdecode(data, cv2.IMREAD_COLOR) for _, data in images_data.items()]
            place_images(full_image, images)
            display_image = cv2.resize(full_image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('', display_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break
        #
        images_data = {}
        n_sample += 1
        log.info('Sample {} saved', n_sample)

log.info("Exiting")
