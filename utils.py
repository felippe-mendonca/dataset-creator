import os
import sys
import cv2
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
        has_frame, frame = self._video_cap.read()
        return (self._current_video_base, n_next_frame, frame)