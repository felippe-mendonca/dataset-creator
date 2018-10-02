import os
import re
import sys
import cv2
import json
import numpy as np
from utils import load_options
from utils import to_labels_array, to_labels_dict
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


def draw_labels(output_image, y_offset, labels, current_pos):
    w, h = output_image.shape[1], output_image.shape[0]
    scale = w / len(labels)
    output_image[h - y_offset:, :, :] = 0

    def draw_rect(x, width, color):
        pt1 = (int(x - width / 2), int(h - y_offset))
        pt2 = (int(x + width / 2), int(h))
        cv2.rectangle(
            img=output_image,
            pt1=pt1,
            pt2=pt2,
            color=color,
            thickness=cv2.FILLED)

    maybe_begins = scale * np.where(labels == 2)[0]
    begins = scale * np.where(labels == 1)[0]
    gestures = scale * np.where(labels == 3)[0]
    ends = scale * np.where(labels == -1)[0]
    for p in maybe_begins:
        draw_rect(p, scale, (0, 255, 0))
    for p in begins:
        draw_rect(p, scale, (255, 0, 0))
    for p in gestures:
        draw_rect(p, scale, (127, 127, 127))
    for p in ends:
        draw_rect(p, scale, (0, 0, 255))
    draw_rect(scale * current_pos, scale, (0, 255, 255))


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
with open('keymap.json', 'r') as f:
    keymap = json.load(f)
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
# put_text(full_image, 'Abcdefghijklmnopqrstuvxywz', x=0, y=0.8 * top_bar_h)

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
    log.info('Loading images from \'{}\'', capture)
    images = load_images(capture, cameras, options.folder)
    log.info('Images loaded')
    n_images = len(images[list(cameras)[0]])
    labels = np.zeros(n_images, dtype=np.int8)
    # check if label file already exists
    labels_file = os.path.join(options.folder, '{}.json'.format(capture))
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = to_labels_array(json.load(f))

    original_labels = np.copy(labels)
    it_image = 0
    update_image, waiting_end, current_begin, current_images = True, False, 0, []
    while True:
        if update_image:
            current_images = [images[camera][it_image] for camera in cameras]
            place_images(full_image, current_images, y_offset=top_bar_h)
            draw_labels(full_image, top_bar_h, labels, it_image)
            cv2.imshow('', cv2.resize(
                full_image, dsize=(0, 0), fx=0.5, fy=0.5))
            update_image = False

        key = cv2.waitKey(0)

        if key == ord(keymap['next_frames']):
            it_image += keymap['big_step']
            it_image = it_image if it_image < n_images else 0
            update_image = True

        if key == ord(keymap['next_frame']):
            it_image += 1
            it_image = it_image if it_image < n_images else 0
            update_image = True

        if key == ord(keymap['previous_frames']):
            it_image -= keymap['big_step']
            it_image = n_images - 1 if it_image < 0 else it_image
            update_image = True

        if key == ord(keymap['previous_frame']):
            it_image -= 1
            it_image = n_images - 1 if it_image < 0 else it_image
            update_image = True

        if key == ord(keymap['begin_label']):
            if labels[it_image] == 0 and not waiting_end:
                labels[it_image] = 2
                current_begin = it_image
                waiting_end = True
                update_image = True
            elif it_image == current_begin and waiting_end:
                labels[it_image] = 0
                waiting_end = False
                update_image = True
            elif (labels[it_image] == -1
                  or labels[it_image] == 3) and not waiting_end:
                previous_begin = np.where(labels[:it_image] == 1)[0][-1]
                it_image = previous_begin
                update_image = True

        if key == ord(keymap['end_label']):
            if labels[it_image] == 0 and waiting_end:
                if it_image > current_begin:
                    labels[current_begin] = 1
                    labels[it_image] = -1
                    labels[current_begin + 1:it_image] = 3
                    waiting_end = False
                    update_image = True
            elif labels[it_image] == -1:
                labels[it_image] = 0
                current_begin = np.where(labels[:it_image] == 1)[0][-1]
                labels[current_begin] = 2
                labels[current_begin + 1:it_image] = 0
                waiting_end = True
                update_image = True
            elif (labels[it_image] == 1
                  or labels[it_image] == 3) and not waiting_end:
                next_end = np.where(labels[it_image:] == -1)[0][0] + it_image
                it_image = next_end
                update_image = True

        if key == ord(keymap['delete_label']):
            if labels[it_image] == 3 and not waiting_end:
                begin = np.where(labels[:it_image] == 1)[0][-1]
                end = np.where(labels[it_image:] == -1)[0][0] + it_image
                labels[begin:end + 1] = 0
                update_image = True

        if key == ord(keymap['save_labels']):
            indexes, counts = np.unique(labels, return_counts=True)
            counts_dict = dict(zip(indexes, counts))
            if not waiting_end and counts_dict[-1] == counts_dict[1]:
                with open(labels_file, 'w') as f:
                    json.dump(to_labels_dict(labels), f, indent=2)
                    log.info("File '{}' saved", labels_file)
                original_labels = labels

        if key == ord(keymap['next_sequence']):
            if np.all(labels == original_labels):
                break
            else:
                log.warn('You have unsaved changes! Save before move to next sequence.')


        if key == ord(keymap['exit']):
            sys.exit(0)

log.info('Exiting')