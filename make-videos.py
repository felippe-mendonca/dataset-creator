import os
import re
import sys
import shutil
from subprocess import Popen, PIPE, STDOUT
from options_pb2 import DatasetCaptureOptions
from is_wire.core import Logger
from google.protobuf.json_format import Parse


def get_person_gesture(folder):
    match = re.search(r'p(\d+)g(\d+)', folder)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)))


log = Logger(name='Capture')

with open('options.json', 'r') as f:
    try:
        options = Parse(f.read(), DatasetCaptureOptions())
    except Exception as ex:
        log.critical('Unable to read \"options.json\". \n{}', ex)
        sys.exit(-1)

if not os.path.exists(options.folder):
    log.critical("Folder '{}' doesn't exist", options.folder)
    sys.exit(-1)

ffmpeg_base_command = "ffmpeg -y -r {fps:.1f} -start_number 0 -i {file_pattern:s} -c:v libx264 -vf fps={fps:.1f} -vf format=rgb24 {video_file:s}"

for root, dirs, files in os.walk(options.folder):
    for exp_folder in dirs:
        pg = get_person_gesture(exp_folder)
        if pg is None:
            continue
        person_id, gesture_id = pg
        sequence_folder = os.path.join(options.folder, exp_folder)
        for camera in options.cameras:
            file_pattern = os.path.join(
                sequence_folder,
                'c{camera_id:02d}s%08d.jpeg'.format(camera_id=camera.id))
            video_file = os.path.join(
                options.folder, 'p{:03d}g{:02d}c{:02d}.mp4'.format(
                    person_id, gesture_id, camera.id))
            ffmpeg_command = ffmpeg_base_command.format(
                fps=camera.config.sampling.frequency.value,
                file_pattern=file_pattern,
                video_file=video_file)
            process = Popen(ffmpeg_command.split(), stdout=PIPE, stderr=STDOUT)
            # with process.stdout as pipe:
                # for line in iter(pipe.readline, b''):
                    # print(line.decode('utf-8').strip())
            if process.wait() == 0:
                log.info("\'{}\' created", video_file)
                # shutil.rmtree(sequence_folder)
            else:
                log.warn("\'{}\' failed", video_file)
    break  # only first folder level
