import os
import cv2

class VideoLoader:
    def __init__(self, filename=None):
        self._vc = None
        self._fps = 0.0
        self._n_frames = 0
        self._frames = []
        self._it = 0
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        self._vc = cv2.VideoCapture(filename)
        self._fps = self._vc.get(cv2.CAP_PROP_FPS)
        self._width = int(self._vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._n_frames = int(self._vc.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frames = []
        while True:
            next_frame_id = int(self._vc.get(cv2.CAP_PROP_POS_FRAMES))
            if next_frame_id == self._n_frames:
                break
            _, frame = self._vc.read()
            self._frames.append(frame)
        self._it = 0

    def fps(self):
        return self._fps
    
    def resolution(self):
        return (self._width, self._height)

    def __next__(self):
        if self._vc is None:
            raise Exception('Load a video file before use next')
        self._it = 0 if self._it == self._n_frames else self._it
        ret_frame = self._frames[self._it]
        self._it += 1 
        return ret_frame


class MultipleVideoLoader:
    def __init__(self, filenames, folder='.'):
        assert (type(filenames) == dict)
        assert (len(filenames) > 0)
        self._filenames = {
            kv[0]: os.path.join(folder, kv[1])
            for kv in filenames.items()
        }
        self._video_captures = {
            src: cv2.VideoCapture(f)
            for src, f in self._filenames.items()
        }
        if not all([vc.isOpened() for vc in self._video_captures.values()]):
            raise Exception("Can't open one of given video files.")
        n_frames = [
            int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            for vc in self._video_captures.values()
        ]
        if not all([nf == n_frames[0] for nf in n_frames]):
            raise Exception('Videos with different number of frames')
        self._n_frames = next(iter(n_frames))
        self._frames = {src: [] for src in self._video_captures.keys()}

    def n_frames(self):
        return self._n_frames

    def n_loaded_frames(self):
        return len(next(iter(self._frames.values())))

    def fps(self):
        return {
            src: vc.get(cv2.CAP_PROP_FPS)
            for src, vc in self._video_captures.items()
        }
    def release_memory(self):
        for src in self._frames.keys():
            del self._frames[src][:]

    def load_next(self):
        next_frame_ids = [
            int(vc.get(cv2.CAP_PROP_POS_FRAMES))
            for vc in self._video_captures.values()
        ]
        if any([n == self._n_frames for n in next_frame_ids]):
            return self.n_loaded_frames()

        frames = {}
        for src, vc in self._video_captures.items():
            _, frame = vc.read()
            self._frames[src].append(frame)

        return self.n_loaded_frames()

    def __getitem__(self, index):
        if not all([
                index < len(video_frames)
                for video_frames in self._frames.values()
        ]):
            return None

        return {
            src: video_frames[index]
            for src, video_frames in self._frames.items()
        }
