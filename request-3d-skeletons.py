import os
import re
import sys
import cv2
import json
import time
import socket
import datetime
from collections import defaultdict
from enum import Enum
from is_wire.core import Channel, Subscription, Message, Logger, ContentType
from is_msgs.image_pb2 import ObjectAnnotations
from utils import load_options, AnnotationsFetcher
from google.protobuf.json_format import MessageToDict

from pprint import pprint

MIN_REQUESTS = 5
MAX_REQUESTS = 10
DEADLINE_SEC = 15.0


class State(Enum):
    MAKE_REQUESTS = 1
    RECV_REPLIES = 2
    CHECK_END_OF_SEQUENCE_AND_SAVE = 3
    CHECK_FOR_TIMEOUTED_REQUESTS = 4
    EXIT = 5


LOCALIZATION_FILE = 'p{:03d}g{:02d}_3d.json'

log = Logger(name='Request3dSkeletons')
options = load_options(print_options=False)

if not os.path.exists(options.folder):
    log.critical("Folder '{}' doesn't exist", options.folder)

files = next(os.walk(options.folder))[2]  # only files from first folder level
annotation_files = list(filter(lambda x: x.endswith('_2d.json'), files))

log.debug('Parsing Annotation Files')
entries = defaultdict(lambda: defaultdict(list))
n_annotations = defaultdict(lambda: defaultdict(dict))
for annotation_file, n in zip(annotation_files, range(len(annotation_files))):

    matches = re.search("p([0-9]{3})g([0-9]{2})c([0-9]{2})_2d.json", annotation_file)
    if matches is None:
        continue
    person_id = int(matches.group(1))
    gesture_id = int(matches.group(2))
    camera_id = int(matches.group(3))
    entries[person_id][gesture_id].append(camera_id)

    annotation_path = os.path.join(options.folder, annotation_file)
    with open(annotation_path, 'r') as f:
        n = len(json.load(f)['annotations'])
        n_annotations[person_id][gesture_id][camera_id] = n

log.debug('Checking if detections files already exists')
cameras = [int(camera_cfg.id) for camera_cfg in options.cameras]
pending_localizations = []
n_localizations = defaultdict(dict)
for person_id, gestures in entries.items():
    for gesture_id, camera_ids in gestures.items():
        if set(camera_ids) != set(cameras):
            log.warn("PERSON_ID: {:03d} GESTURE_ID: {:02d} | Can't find all detections file.",
                     person_id, gesture_id)
            continue

        n_an = list(n_annotations[person_id][gesture_id].values())
        if not all(map(lambda x: x == n_an[0], n_an)):
            log.warn("PERSON_ID: {:03d} GESTURE_ID: {:02d} | Annotations size inconsistent.",
                     person_id, gesture_id)
            continue

        file = os.path.join(options.folder, LOCALIZATION_FILE.format(person_id, gesture_id))
        if os.path.exists(file):
            with open(file, 'r') as f:
                n_loc = len(json.load(f)['localizations'])
            if n_loc == n_an[0]:
                log.info('PERSON_ID: {:03d} GESTURE_ID: {:02d} | Already have localization file.',
                         person_id, gesture_id)
                continue

        n_localizations[person_id][gesture_id] = n_an[0]
        pending_localizations.append({
            'person_id': person_id,
            'gesture_id': gesture_id,
            'n_localizations': n_an[0]
        })

if len(pending_localizations) == 0:
    log.info("Exiting...")
    sys.exit(0)

channel = Channel(options.broker_uri)
subscription = Subscription(channel)

requests = {}
localizations_received = defaultdict(lambda: defaultdict(dict))
state = State.MAKE_REQUESTS
annotations_fetcher = AnnotationsFetcher(
    pending_localizations=pending_localizations, cameras=cameras, base_folder=options.folder)

while True:
    if state == State.MAKE_REQUESTS:

        state = State.RECV_REPLIES
        if len(requests) < MIN_REQUESTS:
            while len(requests) <= MAX_REQUESTS:
                person_id, gesture_id, pos, annotations = annotations_fetcher.next()
                if pos is None:
                    if len(requests) == 0:
                        state = State.EXIT
                    break

                msg = Message(reply_to=subscription, content_type=ContentType.JSON)
                body = json.dumps({'list': annotations}).encode('utf-8')
                msg.body = body
                msg.timeout = DEADLINE_SEC
                channel.publish(msg, topic='SkeletonsGrouper.Localize')
                requests[msg.correlation_id] = {
                    'body': body,
                    'person_id': person_id,
                    'gesture_id': gesture_id,
                    'pos': pos,
                    'requested_at': time.time()
                }
        continue

    elif state == State.RECV_REPLIES:

        try:
            msg = channel.consume(timeout=1.0)
            if msg.status.ok():
                localizations = msg.unpack(ObjectAnnotations)
                cid = msg.correlation_id
                if cid in requests:
                    person_id = requests[cid]['person_id']
                    gesture_id = requests[cid]['gesture_id']
                    pos = requests[cid]['pos']
                    localizations_received[person_id][gesture_id][pos] = MessageToDict(
                        localizations,
                        preserving_proto_field_name=True,
                        including_default_value_fields=True)
                    del requests[cid]

            state = State.CHECK_END_OF_SEQUENCE_AND_SAVE
        except socket.timeout:
            state = State.CHECK_FOR_TIMEOUTED_REQUESTS
        continue

    elif state == State.CHECK_END_OF_SEQUENCE_AND_SAVE:

        done_sequences = []
        for person_id, gestures in localizations_received.items():
            for gesture_id, localizations_dict in gestures.items():
                if len(localizations_dict) < n_localizations[person_id][gesture_id]:
                    continue

                output_localizations = {
                    'localizations': [x[1] for x in sorted(localizations_dict.items())],
                    'created_at': datetime.datetime.now().isoformat()
                }
                filename = 'p{:03d}g{:02d}_3d.json'.format(person_id, gesture_id)
                filepath = os.path.join(options.folder, filename)
                with open(filepath, 'w') as f:
                    json.dump(output_localizations, f, indent=2)

                done_sequences.append((person_id, gesture_id))
                log.info('{} has been saved.', filename)

        for person_id, gesture_id in done_sequences:
            del localizations_received[person_id][gesture_id]

        state = State.CHECK_FOR_TIMEOUTED_REQUESTS
        continue

    elif state == State.CHECK_FOR_TIMEOUTED_REQUESTS:

        new_requests = {}
        for cid in list(requests.keys()):
            request = requests[cid]
            if (request['requested_at'] + DEADLINE_SEC) > time.time():
                continue
            msg = Message(reply_to=subscription, content_type=ContentType.JSON)
            msg.body = request['body']
            msg.timeout = DEADLINE_SEC
            channel.publish(msg, topic='SkeletonsGrouper.Localize')
            new_requests[msg.correlation_id] = {
                'body': request['body'],
                'person_id': request['gesture_id'],
                'gesture_id': request['gesture_id'],
                'pos': request['pos'],
                'requested_at': time.time()
            }
            del requests[cid]
            log.warn("Message '{}' timeouted. Sending another request.", cid)

        requests.update(new_requests)
        state = State.MAKE_REQUESTS
        continue

    elif state == State.EXIT:

        log.info("Exiting...")
        sys.exit(-1)

    else:

        state = State.MAKE_REQUESTS
        continue
