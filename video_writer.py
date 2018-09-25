import cv2
import numpy as np
from queue import Queue
from threading import Thread

fourcc = 0x00000021  # H264 codec code


def to_cv_mat(pb_image):
    data = np.fromstring(pb_image.data, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class VideoWriter:
    def __init__(self, queue_size=500):
        self._video_writers = {}
        self._queue = Queue(maxsize=queue_size)
        self._writer_thread = Thread(target=self._writer)
        self._writer_thread.daemon = True
        self._writer_thread.start()

    def add_camera(self, camera_id, filename, fps, resolution):
        self._video_writers[camera_id] = cv2.VideoWriter(
            filename=filename, fourcc=fourcc, fps=fps, frameSize=resolution)

    def _writer(self):
        while True:
            camera_id, image = self._queue.get()
            if camera_id not in self._video_writers:
                continue
            if image is None:
                self._video_writers[camera_id].release()
                self._queue.task_done()
                if all([not vw.isOpened() for _, vw in self._video_writers.items()]):
                    break
            else:
                self._video_writers[camera_id].write(image)
                self._queue.task_done()

    def write(self, camera_id, image):
        self._queue.put((camera_id, image))

    def join(self):
        self._queue.join()
