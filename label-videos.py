import os
import re
import sys
import cv2
import numpy as np
from utils import load_options
from is_wire.core import Logger
from collections import defaultdict
import time


def load_images(capture, cameras, folder):
    video_captures = {}
    for camera in cameras:
        video_file = '{:s}c{:02d}.mp4'.format(capture, camera)
        video_path = os.path.join(folder, video_file)
        video_captures[camera] = cv2.VideoCapture(video_path)

    n_frames = [
        int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        for vc in video_captures.values()
    ]

    if not all([nf == n_frames[0] for nf in n_frames]):
        return None

    images = defaultdict(list)
    for camera, video_capture in video_captures.items():
        while True:
            has_frame, frame = video_capture.read()
            if not has_frame:
                break
            images[camera].append(frame)

    return images


def place_images(output_image, images, x_offset=0, y_offset=0):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0 + y_offset:h + y_offset, 0 + x_offset:w +
                 x_offset, :] = images[0]
    output_image[0 + y_offset:h + y_offset, w + x_offset:2 * w +
                 x_offset, :] = images[1]
    output_image[h + y_offset:2 * h + y_offset, 0 + x_offset:w +
                 x_offset, :] = images[2]
    output_image[h + y_offset:2 * h + y_offset, w + x_offset:2 * w +
                 x_offset, :] = images[3]


def put_text(image,
             text,
             x,
             y,
             color=(255, 255, 255),
             font_scale=1.5,
             thickness=2):
    cv2.putText(
        img=image,
        text=text,
        org=(int(x), int(y)),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=font_scale,
        color=color,
        thickness=thickness)


log = Logger(name='LabelVideos')
options = load_options(print_options=False)

if not os.path.exists(options.folder):
    log.critical("Folder '{}' doesn't exist", options.folder)
    sys.exit(-1)

bottom_bar_h = 50
top_bar_h = 75
height = 2 * options.cameras[0].config.image.resolution.height
width = 2 * options.cameras[0].config.image.resolution.width
size = (height + bottom_bar_h + top_bar_h, width, 3)
full_image = np.zeros(size, dtype=np.uint8)
full_image[(top_bar_h - 5):(top_bar_h - 1), :, :] = 255
full_image[top_bar_h + height:(top_bar_h + height + 5), :, :] = 255
put_text(full_image, 'Abcdefghijklmnopqrstuvxywz', x=0, y=0.8 * top_bar_h)

files = next(os.walk(options.folder))[2]  # only files from first folder level
video_files = list(filter(lambda x: x.endswith('.mp4'), files))

captures = defaultdict(set)
for video_file in video_files:
    matches = re.search(r'(p\d+g\d+)c(\d+).mp4$', video_file)
    if matches is None:
        continue
    cap_name, camera = matches.group(1), int(matches.group(2))
    captures[cap_name].add(camera)

for capture, cameras in captures.items():
    images = load_images(capture, cameras, options.folder)
    n_images = len(images[list(cameras)[0]])
    it_image = 0
    update_image, current_images = True, []
    while True:
        if update_image:
            current_images = [images[camera][it_image] for camera in cameras]
            place_images(full_image, current_images, y_offset=top_bar_h)
            cv2.imshow('', cv2.resize(full_image, dsize=(0,0), fx=0.5, fy=0.5))
            update_image = False

        key = cv2.waitKey(0)
        if key == ord('l') or key == ord('L'):
            it_image += 1
            it_image = it_image if it_image < n_images else 0
            update_image = True
        if key == ord('j') or key == ord('J'):
            it_image -= 1
            it_image = n_images - 1 if it_image < 0 else it_image
            update_image = True