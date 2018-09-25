from is_wire.core import Channel, Message, Logger
from is_msgs.image_pb2 import Image
from options_pb2 import DatasetCaptureOptions
from google.protobuf.json_format import Parse
import cv2
import time

cameras = [0, 1, 2, 3]
n_samples = 91
period_ms = 100
drift_ms = 10
log = Logger(name='Capture')

with open('options.json', 'r') as f:
    try:
        options = Parse(f.read(), DatasetCaptureOptions())
        log.info('Options:\n{}', options)
    except Exception as ex:
        log.critical('Unable to read \"options.json\". \n{}', ex)

image_data = {}
for camera in cameras:
    image = cv2.imread('{}.jpeg'.format(camera), cv2.IMREAD_COLOR)
    image_data[camera] = cv2.imencode('.jpeg', image)[1].tobytes()

channel = Channel(options.broker_uri)

while True:
    for sample in range(n_samples):
        t0 = time.time()
        log.info("Sample: {}", sample)
        for camera in cameras:
            filename = '/home/felippe/aaaaa/cam{}_{:04d}.jpeg'.format(
                camera + 1, sample)
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            data = cv2.imencode('.jpeg', image)[1].tobytes()
            pb_image = Image(data=data)
            msg = Message(content=pb_image)
            topic = 'CameraGateway.{}.Frame'.format(camera)
            channel.publish(msg, topic=topic)
            time.sleep(drift_ms / 1000.0)
        tf = time.time()
        dt = period_ms / 1000.0 - (tf - t0)
        if dt > 0:
            time.sleep(dt)
