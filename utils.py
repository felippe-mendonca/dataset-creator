import os
import sys
import cv2
import json
import numpy as np
from options_pb2 import DatasetCaptureOptions
from google.protobuf.json_format import Parse
from is_wire.core import Logger
from is_msgs.image_pb2 import Image


def load_options(print_options=True):
    log = Logger(name='LoadOptions')
    with open('options.json', 'r') as f:
        try:
            options = Parse(f.read(), DatasetCaptureOptions())
            if print_options:
                log.info('Options:\n{}', options)
            return options
        except Exception as ex:
            log.critical('Unable to read \"options.json\". \n{}', ex)
            sys.exit(-1)


def make_pb_image(input_image, encode_format='.jpeg', compression_level=0.9):
    if isinstance(input_image, np.ndarray):
        if encode_format == '.jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
        elif encode_format == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
        else:
            return Image()
        cimage = cv2.imencode(ext=encode_format, img=input_image, params=params)
        return Image(data=cimage[1].tobytes())
    elif isinstance(input_image, Image):
        return input_image
    else:
        return Image()


def to_labels_array(labels_dict):
    labels = np.zeros(labels_dict['n_samples'])
    for label in labels_dict['labels']:
        begin, end = label['begin'], label['end']
        labels[begin] = 1
        labels[end] = -1
        labels[begin + 1:end] = 3
    return labels


def to_labels_dict(labels_array):
    labels = {'n_samples': labels_array.size, 'labels': []}
    begins = np.where(labels_array == 1)[0]
    ends = np.where(labels_array == -1)[0]
    if (begins.size != ends.size) and begins.size > 0 and ends.size > 0:
        return labels
    diff = ends - begins
    if np.any(diff < 1):
        return labels
    for begin, end in zip(begins.tolist(), ends.tolist()):
        labels['labels'].append({'begin': begin, 'end': end})
    return labels


class FrameVideoFetcher:
    def __init__(self, video_files, base_folder):
        self._video_files = video_files
        self._it_videos = iter(self._video_files)
        self._base_folder = base_folder
        self._video_cap = cv2.VideoCapture()
        self._current_video_base = ''

    def next(self):
        n_frames = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_next_frame = int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not self._video_cap.isOpened() or n_frames == n_next_frame:
            try:
                while True:
                    video_file = next(self._it_videos)
                    video_path = os.path.join(self._base_folder, video_file)
                    if self._video_cap.open(video_path):
                        break
                self._current_video_base = video_file.split('.')[0]
            except:
                return '', 0, None

        n_next_frame = int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        _, frame = self._video_cap.read()


class AnnotationsFetcher:
    def __init__(self, pending_localizations, cameras, base_folder, fix_frame_id=True):
        self._pending_localizations = pending_localizations
        self._cameras = cameras
        self._base_folder = base_folder
        self._fix_frame_id = fix_frame_id
        self._localizations_it = iter(self._pending_localizations)
        self._annotation_pos = 0
        self._n_annotations = 0
        self._fwd_annotations = True
        self._current_annotations = {}
        self._current_person_id = None
        self._current_gesture_id = None

    def next(self):
        if self._fwd_annotations:
            try:
                pending_localization = next(self._localizations_it)
                self._current_person_id = pending_localization['person_id']
                self._current_gesture_id = pending_localization['gesture_id']
                self._n_annotations = pending_localization['n_localizations']
                for camera in self._cameras:
                    filename = 'p{:03d}g{:02d}c{:02d}_2d.json'.format(
                        self._current_person_id, self._current_gesture_id, camera)
                    filepath = os.path.join(self._base_folder, filename)
                    with open(filepath, 'r') as f:
                        annotations = json.load(f)['annotations']
                        if self._fix_frame_id:
                            for annotation in annotations:
                                annotation['frame_id'] = camera
                        self._current_annotations[camera] = annotations

                self._annotation_pos = 0
                self._fwd_annotations = False
            except StopIteration:
                return None, None, None, None

        annotations = [
            self._current_annotations[camera][self._annotation_pos] for camera in self._cameras
        ]
        pos = self._annotation_pos

        self._annotation_pos += 1
        self._fwd_annotations = self._annotation_pos == self._n_annotations
        return self._current_person_id, self._current_gesture_id, pos, annotations
