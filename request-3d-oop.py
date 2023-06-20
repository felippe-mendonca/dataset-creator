import datetime
import json
import os
import re
import socket
import time
from collections import defaultdict
from enum import Enum
from glob import glob

from google.protobuf.json_format import MessageToDict
from is_msgs.image_pb2 import ObjectAnnotations
from is_wire.core import Channel, ContentType, Logger, Message, Subscription
from utils import AnnotationsFetcher, load_options

MIN_REQUESTS = 50  # Número mínimo de solicitações
MAX_REQUESTS = 1000  # Número máximo de solicitações
DEADLINE_SEC = 5.0  # Prazo limite em segundos
JSON2D_REGEX = 'p([0-9]{3})g([0-9]{2})c([0-9]{2})_2d.json'
JSON3D_FORMAT = 'p{:03d}g{:02d}_3d.json'
# Enumeração para representar os diferentes estados do programa
class State(Enum):
    MAKE_REQUESTS = 1
    RECV_REPLIES = 2
    CHECK_END_OF_SEQUENCE_AND_SAVE = 3
    CHECK_FOR_TIMEDOUT_REQUESTS = 4
    EXIT = 5


class Request3D:
    def __init__(self):
        self.log = Logger(name='Request3dSkeletons')

        self.state = None
        self.pending_localizations:list = [] # Preenche em self.check_for_detection_files()

        self.requests:dict = {} # Preenche em self._make_requests()
        self.num_localizations:defaultdict = defaultdict(dict) # Preenche em self.check_for_detection_files()
        self.localizations_received:defaultdict = defaultdict(lambda: defaultdict(dict)) # Preenche em self._recv_replies()


        self.options = self._get_options()
        self.channel, self.subscription = self._get_communication()
        self.annotation_files = self._get_annotation_files()
        self.person_gesture_camera_dict, self.quantity_of_annotations = self._get_person_gesture_camera()
        self.cameras_id_list:list = self._get_cameras_id_list()

        self.check_for_detection_files()

        self.annotations_fetcher:AnnotationsFetcher = AnnotationsFetcher(pending_localizations=self.pending_localizations, cameras=self.cameras_id_list, base_folder=self.options.folder)
    
    def _get_cameras_id_list(self):
        return [int(camera_cfg.id) for camera_cfg in self.options.cameras]
    
    def _get_communication(self):
        # Comunicação
        channel = Channel(self.options.broker_uri)
        subscription = Subscription(channel)

        return channel, subscription

    def _get_annotation_files(self):
        self.log.debug('Getting annotation files')
        annotation_files = list(map(lambda s: s.replace(self.options.folder+'/',''),glob(os.path.join(self.options.folder, '*_2d.json'))))
        return annotation_files

    def _get_options(self):
        options = load_options(print_options=False)
        if not os.path.exists(options.folder):
            self.log.critical("Folder '{}' doesn't exist", options.folder)
        return options
    
    def _get_person_gesture_camera(self):
        
        # Dicionário de dicionários de listas
        person_gesture_camera_dict:defaultdict = defaultdict(lambda: defaultdict(list))
        # Dicionário de dicionários de dicionários
        quantity_of_annotations = defaultdict(lambda: defaultdict(dict))
        for file in self.annotation_files:
            # Extrai os IDs de pessoa, gesto e câmera do nome do arquivo de anotação
            matches = re.search(JSON2D_REGEX,  file)
            if matches is None:
                continue

            person_id = int(matches.group(1))
            gesture_id = int(matches.group(2))
            camera_id = int(matches.group(3))

            # Adiciona as informações de pessoa, gesto e câmera ao dicionário de dicionários de listas
            person_gesture_camera_dict[person_id][gesture_id].append(camera_id)

            # Lê o arquivo de anotação e conta o número de anotações
            annotation_path:str = os.path.join(self.options.folder,  file)

            with open(annotation_path) as f:
                len_annotations = len(json.load(f)['annotations'])
                quantity_of_annotations[person_id][gesture_id][camera_id] = len_annotations
        return person_gesture_camera_dict, quantity_of_annotations
    
    def check_for_detection_files(self):
        """Checa se o file de localização 3D já existe e se o número de anotações é igual ao número de localizações.
        """

        for person_id, gestures in self.person_gesture_camera_dict.items():
            for gesture_id, camera_ids in gestures.items():
                file = os.path.join(self.options.folder, JSON3D_FORMAT.format(person_id, gesture_id))
                
                number_of_annotations:list = list(self.quantity_of_annotations[person_id][gesture_id].values())


                if set(camera_ids) != set(self.cameras_id_list):
                    self.log.warn("PERSON_ID: {:03d} GESTURE_ID: {:02d} | Can't find all detections file.",
                            person_id, gesture_id)
                    continue


                if not all(map(lambda x: x == number_of_annotations[0], number_of_annotations)):
                    self.log.warn("PERSON_ID: {:03d} GESTURE_ID: {:02d} | Annotations size inconsistent.",
                            person_id, gesture_id)
                    continue

                
                if os.path.exists(file):
                    with open(file, 'r') as f:
                        number_of_localizations = len(json.load(f)['localizations'])
                    if number_of_localizations == number_of_annotations[0]:
                        self.log.info('PERSON_ID: {:03d} GESTURE_ID: {:02d} | Already have localization file.',
                                person_id, gesture_id)
                        continue

                self.num_localizations[person_id][gesture_id] = number_of_annotations[0]
                self.pending_localizations.append({
                    'person_id': person_id,
                    'gesture_id': gesture_id,
                    'n_localizations': number_of_annotations[0]
                })

        if not self.pending_localizations:
            self.log.critical("Exiting... No pending localizations.")

    def _publish(self,msg,topic='SkeletonsGrouper.Localize'):
        self.channel.publish(msg, topic=topic)

    def run(self):
        """ Executa o while True loop de requests
        """
        self.state = State.MAKE_REQUESTS

        while self.state != State.EXIT:
            if self.state == State.MAKE_REQUESTS:
                self._make_requests()
            elif self.state == State.RECV_REPLIES:
                self._recv_replies()
            elif self.state == State.CHECK_FOR_TIMEDOUT_REQUESTS:
                self._check_for_timed_out_requests()
            elif self.state == State.CHECK_END_OF_SEQUENCE_AND_SAVE:
                self._check_end_of_sequence_and_save()

    def _make_requests(self):
        

        self.state = State.RECV_REPLIES
            
        if len(self.requests) < MIN_REQUESTS:
            while len(self.requests) <= MAX_REQUESTS:
                person_id, gesture_id, pos, annotations = self.annotations_fetcher.next()
                if pos is None:
                    if not self.requests:
                        self.state = State.EXIT
                    break

                msg = Message(reply_to=self.subscription, content_type=ContentType.JSON)
                body = json.dumps({'list': annotations}).encode('utf-8')
                msg.body = body
                msg.timeout = DEADLINE_SEC
                
                self._publish(msg)

                self.requests[msg.correlation_id] = {
                    'body': body,
                    'person_id': person_id,
                    'gesture_id': gesture_id,
                    'pos': pos,
                    'requested_at': time.time()
                }

    def _recv_replies(self):
        """_summary_
        """        
        try:
            msg = self.channel.consume(timeout=1.0)
            if msg.status.ok():
                localizations = msg.unpack(ObjectAnnotations)
                correlation_id = msg.correlation_id

                if correlation_id in self.requests:
                    person_id = self.requests[correlation_id]['person_id']
                    gesture_id = self.requests[correlation_id]['gesture_id']
                    pos = self.requests[correlation_id]['pos']
                    self.localizations_received[person_id][gesture_id][pos] = MessageToDict(
                        localizations,
                        preserving_proto_field_name=True,
                        including_default_value_fields=True)
                    self.requests.pop(correlation_id)

        except socket.timeout:
            self.state = State.CHECK_FOR_TIMEDOUT_REQUESTS
        else:
            self.state = State.CHECK_END_OF_SEQUENCE_AND_SAVE

    def _check_for_timed_out_requests(self):
        """_summary_
        """        
        new_requests = {}

        for cid in list(self.requests.keys()):
            request = self.requests[cid]
            if (request['requested_at'] + DEADLINE_SEC) > time.time():
                continue
            msg = Message(reply_to=self.subscription, content_type=ContentType.JSON)
            msg.body = request['body']
            msg.timeout = DEADLINE_SEC
            
            self._publish(msg)

            new_requests[msg.correlation_id] = {
                'body': request['body'],
                'person_id': request['gesture_id'],
                'gesture_id': request['gesture_id'],
                'pos': request['pos'],
                'requested_at': time.time()
            }
            del self.requests[cid]
            self.log.warn("Message '{}' timed out. Sending another request.", cid)

        self.requests.update(new_requests)
        
        self.state = State.MAKE_REQUESTS

    def _check_end_of_sequence_and_save(self):
        """_summary_
        """        
        done_sequences:list = []
        # Error para p001g03 e g02???
        for person_id, gestures in self.localizations_received.items():
            for gesture_id, localizations_dict in gestures.items():
                try:
                    print(person_id, gesture_id, len(localizations_dict), self.num_localizations[person_id][gesture_id])
                    if len(localizations_dict) < self.num_localizations[person_id][gesture_id]:
                        continue
                except KeyError:
                    self.log.warn(f'KeyError: PERSON_ID: {person_id:03d} GESTURE_ID: {gesture_id:02d}')
                    continue

                output_localizations = {
                    'localizations': [x[1] for x in sorted(localizations_dict.items())],
                    'created_at': datetime.datetime.now().isoformat()
                }
                filename = JSON3D_FORMAT.format(person_id, gesture_id)
                filepath = os.path.join(self.options.folder, filename)
                with open(filepath, 'w') as f:
                    json.dump(output_localizations, f, indent=2)

                done_sequences.append((person_id, gesture_id))

                self.log.info('Saved: PERSON_ID: {:03d} GESTURE_ID: {:02d}',
                        person_id, gesture_id)

        # Remove as sequências que já foram salvas
        for person_id, gesture_id in done_sequences:
            del self.localizations_received[person_id][gesture_id]

        self.state = State.CHECK_FOR_TIMEDOUT_REQUESTS

if __name__ == '__main__':
    request3d = Request3D()
    request3d.run()