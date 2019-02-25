import datetime
import traceback
import os
import json
import uuid
import hashlib
import queue
import tornado.web
import psycopg2
import operator
from tornado.escape import json_encode, json_decode
from .db import PostgresClient
from .entities import Person, Face
from .stages import *
from surround import Surround, Config
from .utils import distance

# Get time for uptime calculation.
START_TIME = datetime.datetime.now()

# Load config file.
CONFIG = Config()
CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
CONFIG.read_config_files([CONFIG_PATH])

POSTGRES_CLIENT = PostgresClient(CONFIG["postgres"]["db"], CONFIG["postgres"]["user"], CONFIG["postgres"]["host"], CONFIG["postgres"]["port"], CONFIG["postgres"]["password"])

SURROUND = Surround([PhotoExtraction(), DownsampleImage(), RotateImage(), ImageTooDark(), DetectAndAlignFaces(), LargestFace(), FaceTooBlurry(), ExtractEncodingsResNet1()])
SURROUND.set_config(CONFIG)
SURROUND.init_stages()


class FaceRecognitionWebApplication(tornado.web.Application):
    def __init__(self, **kwargs):
        kwargs["handlers"] = [
            (r"/", MainHandler),
            (r"/info", InfoHandler),
            (r"/persons/photo-search", PhotoSearchHandler),
            (r"/persons/encoding-search", EncodingSearchHandler),
            (r"/persons/(?P<person_id>.*)/faces/(?P<face_id>.*)", FaceHandler),
            (r"/persons/(?P<person_id>.*)/faces", FaceHandler),
            (r"/persons/(?P<person_id>.*)", PersonHandler),
            (r"/persons", PersonCollectionHandler),
            (r"/faces", FaceCollectionHandler),
            (r"/encode", AdHocEncodingHandler),
        ]
        super().__init__(**kwargs)


class FaceRecognitionError(tornado.web.HTTPError):
    """
    Exception class for any application-specific errors that occur.
    Usage: `raise FaceRecognitionError(reason="Image too blurry.", status_code=400)`
    """
    pass


class FaceRecognitionWebHandler(tornado.web.RequestHandler):
    """
    Base class for web handlers that will accept requests with application/json content-type.
    """
    def prepare(self):
        """
        Parses the request body and populates a json_body variable.
        """
        self.set_header("Content-Type", "application/json")
        self.json_body = dict()
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            if len(self.request.body) > 0:
                self.json_body = json_decode(self.request.body.decode("utf-8"))

    def write_error(self, status_code, **kwargs):
        self.set_header("Content-Type", "application/json")
        if self.settings.get("debug") and "exc_info" in kwargs:
            # In debug mode, try to send a traceback.
            lines = []
            for line in traceback.format_exception(*kwargs["exc_info"]):
                lines.append(line)
            self.finish(dict(
                error=dict(
                    code=status_code,
                    message=self._reason,
                    traceback=lines
                ))
            )
        else:
            self.finish(dict(
                error=dict(
                    code=status_code,
                    message=self._reason
                ))
            )


class MainHandler(FaceRecognitionWebHandler):
    def get(self):
        self.write(dict(
            app="A2I2 Face Recognition",
            version="0.1",  # @TODO: Link up to actual version
            uptime=str(datetime.datetime.now() - START_TIME)
        ))


class InfoHandler(FaceRecognitionWebHandler):
    def get(self):
        routes = dict(routes=[
            dict(request="POST /persons",                           description="Create a person"),
            dict(request="GET /persons",                            description="Get all persons"),
            dict(request="GET /persons?name=name",                  description="Search for a person by name"),
            dict(request="GET /persons/:id",                        description="Get a single person by ID"),
            dict(request="GET /persons/:id/faces",                  description="Get all face encodings for a person"),
            dict(request="POST /persons/:id/faces",                 description="Add a face to a person"),
            dict(request="GET /persons/:id/faces/:id",              description="Get a single face for a person by ID"),
            dict(request="DELETE /persons/:id/faces/:id",           description="Delete a face for a person"),
            dict(request="DELETE /persons/:id/faces",               description="Delete all faces for a person"),
            dict(request="DELETE /persons",                         description="Delete all persons"),
            dict(request="POST /persons/photo-search",              description="Search for a person using a photo"),
            dict(request="POST /persons/photo-search?nearest=N",    description="Get a list of the nearest people (in order of confidence) using a photo, up to a maximum of N people"),
            dict(request="POST /persons/encoding-search",           description="Search for a person using an encoding"),
            dict(request="POST /persons/encoding-search?nearest=N", description="Get a list of the nearest people (in order of confidence) using a photo, up to a maximum of N people"),
            dict(request="GET /faces",                              description="Get all face encodings"),
            dict(request="DELETE /faces",                           description="Delete all face encodings"),
            dict(request="POST /encode",                            description="Perform a once-off encoding (don't save anything)")
        ])

        self.write(routes)


class PersonCollectionHandler(FaceRecognitionWebHandler):
    def get(self):
        name = self.get_query_argument("name", None)
        people = []
        if name is not None:
            people = POSTGRES_CLIENT.find_persons_by_name(name)
        else:
            people = POSTGRES_CLIENT.find_all_persons()

        self.write(json_encode([person.__dict__ for person in people]))

    def post(self):
        try:
            person = Person(id=None, name=self.json_body["name"])
        except KeyError as e:
            raise FaceRecognitionError(reason="Missing field: " + str(e), status_code=400)

        result = POSTGRES_CLIENT.create_person(person)
        self.set_status(201)
        self.write(result.__dict__)

    def delete(self):
        POSTGRES_CLIENT.delete_all_persons()
        self.set_status(200)


class PersonHandler(FaceRecognitionWebHandler):
    def _person_uuid(self, person_id):
        try:
            return uuid.UUID(person_id)
        except ValueError as e:
            raise FaceRecognitionError(reason=str(e), status_code=400)

    def get(self, person_id):
        result = POSTGRES_CLIENT.find_person_by_id(self._person_uuid(person_id))
        self.write(result.__dict__)

    def delete(self, person_id):
        POSTGRES_CLIENT.delete_person_by_id(self._person_uuid(person_id))
        self.set_status(200)


class FaceCollectionHandler(FaceRecognitionWebHandler):
    def get(self):
        faces = POSTGRES_CLIENT.find_all_faces()
        self.write(json_encode([face.serializable() for face in faces]))

    def delete(self):
        POSTGRES_CLIENT.delete_all_faces()
        self.set_status(200)


class PipelineHandler(FaceRecognitionWebHandler):
    def _pipeline_data_from_photo(self):
        try:
            file_info = self.request.files["file"][0]
        except KeyError:
            raise FaceRecognitionError(reason="No file uploaded via 'file' form-data field")
        return PipelineData(image_filename=file_info["filename"], image_bytes=file_info["body"])

    def _run_pipeline(self, data, single_face=True):
        SURROUND.process(data)

        if data.error is not None:
            raise FaceRecognitionError(reason=str(data.error["error"]), status_code=400)
        if len(data.face_encodings) == 0:
            raise FaceRecognitionError(reason="No face found", status_code=400)
        if single_face and len(data.face_encodings) > 1:
            raise FaceRecognitionError(reason="More than one face in registration photo", status_code=400)

        faces = []
        for i in range(len(data.face_bounding_boxes)):
            faces.append(Face(
                id=None,
                person_id=None,
                encoding=data.face_encodings[i],
                photo_md5=hashlib.md5(data.image_bytes).hexdigest(),
                photo_filename=data.image_filename,
                box_x1=data.face_filtered_boxes[i][0],
                box_x2=data.face_filtered_boxes[i][2],
                box_y1=data.face_filtered_boxes[i][1],
                box_y2=data.face_filtered_boxes[i][3],
                encoder_version="v1",
                encoder_batch_id=uuid.uuid4()
            ))

        if single_face:
            return faces[0]  # We checked above, so we know there's at least one face
        else:
            return faces


class FaceHandler(PipelineHandler):
    def get(self, person_id, face_id=None):
        faces = []
        if face_id is not None:
            face = POSTGRES_CLIENT.find_person_face(person_id, face_id)
            self.write(face.serializable())
        else:
            faces = POSTGRES_CLIENT.find_person_faces(person_id)
            self.write(json_encode([face.serializable() for face in faces]))

    def post(self, person_id):
        data = self._pipeline_data_from_photo()
        face = self._run_pipeline(data, single_face=True)
        face.person_id = person_id

        try:
            result = POSTGRES_CLIENT.create_face_for_person(person_id, face)
        except psycopg2.IntegrityError:
            raise FaceRecognitionError(reason="Person already has an identical face encoding", status_code=400)

        self.set_status(201)
        self.write(result.serializable())

    def delete(self, person_id, face_id=None):
        if face_id is not None:
            POSTGRES_CLIENT.delete_face_for_person(person_id, face_id)
        else:
            POSTGRES_CLIENT.delete_faces_for_person(person_id)


class SearchHandler(PipelineHandler):
    def prepare(self):
        super().prepare()
        nearest = self.get_query_argument("nearest", None)
        if nearest is not None:
            try:
                nearest = int(nearest)
                if nearest <= 0:
                    raise ValueError
            except ValueError:
                raise FaceRecognitionError(reason="'nearest' must be a positive integer", status_code=400)

        self.nearest = nearest

    def _closest_people(self, encoding, limit=None):
        # @TODO: Naive approach used here for v1. Will need optimising for scale.
        # Suggest adding the concept of 'groups' to limit the search space.
        all_faces = POSTGRES_CLIENT.find_all_faces()
        if len(all_faces) == 0:
            raise FaceRecognitionError(reason="No people in the database", status_code=404)

        distances = dict()
        for other_face in all_faces:
            dist = distance(encoding, other_face.encoding)
            distances[str(other_face.person_id)] = dist

        closest = dict()
        for i, (id, dist) in enumerate(sorted(distances.items(), key=operator.itemgetter(1))[:limit]):
            person = POSTGRES_CLIENT.find_person_by_id(uuid.UUID(id))
            closest[str(i)] = dict(person=person.__dict__, distance=dist)

        return closest


class PhotoSearchHandler(SearchHandler):
    def post(self):
        data = self._pipeline_data_from_photo()
        face = self._run_pipeline(data, single_face=True)

        if self.nearest is None:
            closest = self._closest_people(face.encoding, limit=1)
            self.write(closest["0"])
        else:
            closest = self._closest_people(face.encoding, limit=self.nearest)
            self.write(closest)


class EncodingSearchHandler(SearchHandler):
    def post(self):
        try:
            encoding = self.json_body["encoding"]
        except KeyError as e:
            raise FaceRecognitionError(reason="Missing field: " + str(e), status_code=400)

        if not isinstance(encoding, list):
            raise FaceRecognitionError(reason="Encoding must be a list of 128 doubles", status_code=400)

        if self.nearest is None:
            closest = self._closest_people(encoding, limit=1)
            self.write(closest["0"])
        else:
            closest = self._closest_people(encoding, limit=self.nearest)
            self.write(closest)


class AdHocEncodingHandler(PipelineHandler):
    def post(self):
        data = self._pipeline_data_from_photo()
        face = self._run_pipeline(data, single_face=True)

        self.write(face.serializable())
