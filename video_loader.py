import os
import cv2


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
