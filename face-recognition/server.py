import datetime
import traceback
import os
import json
import uuid
import hashlib
import tornado.web
import psycopg2
from tornado.escape import json_encode, json_decode
from .db import PostgresClient
from .entities import Person, Face
from .stages import *
from surround import Surround, Config


START_TIME = datetime.datetime.now()
POSTGRES_CLIENT = PostgresClient("face_recognition", "postgres", "localhost", 5432, "postgres")
SURROUND = Surround([PhotoExtraction(), DownsampleImage(), RotateImage(), ImageTooDark(), DetectAndAlignFaces(), LargestFace(), FaceTooBlurry(), ExtractEncodingsResNet1()])
CONFIG = Config()
CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
CONFIG.read_config_files([CONFIG_PATH])
SURROUND.set_config(CONFIG)
SURROUND.init_stages()


class SurroundWebApplication(tornado.web.Application):
    def __init__(self, **kwargs):
        kwargs["handlers"] = [
            (r"/", MainHandler),
            (r"/info", InfoHandler),
            (r"/persons/(?P<person_id>.*)/faces/(?P<face_id>.*)", FaceHandler),
            (r"/persons/(?P<person_id>.*)/faces", FaceHandler),
            (r"/persons/(?P<person_id>.*)", PersonHandler),
            (r"/persons/photo-search", PhotoSearchHandler),
            (r"/persons/encoding-search", EncodingSearchHandler),
            (r"/persons", PersonCollectionHandler),
            (r"/faces", FaceCollectionHandler),
            (r"/encode", AdHocEncodingHandler),
        ]
        super().__init__(**kwargs)


class SurroundWebError(tornado.web.HTTPError):
    """
    Exception class for any application-specific errors that occur.
    Usage: `raise SurroundWebError(reason="Wrong age value.", status_code=400)`
    """
    pass


class SurroundHandler(tornado.web.RequestHandler):
    """
    Base class for web handlers that will accept requests with application/json content-type.
    """
    def prepare(self):
        """
        Parses the request body and populates a json_body variable.
        """
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
            self.finish(json_encode(dict(
                error=dict(
                    code=status_code,
                    message=self._reason,
                    traceback=lines
                ))
            ))
        else:
            self.finish(json_encode(dict(
                error=dict(
                    code=status_code,
                    message=self._reason
                ))
            ))


class MainHandler(SurroundHandler):
    def get(self):
        self.write(json_encode(dict(
            app="A2I2 Face Recognition",
            version="0.1",
            uptime=str(datetime.datetime.now() - START_TIME)
        )))


class InfoHandler(SurroundHandler):
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

        self.write(json_encode(routes))


class PersonCollectionHandler(SurroundHandler):
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
            raise SurroundWebError(reason="Missing field: " + str(e), status_code=400)

        result = POSTGRES_CLIENT.create_person(person)
        self.set_status(201)
        self.write(json_encode(result.__dict__))

    def delete(self):
        POSTGRES_CLIENT.delete_all_persons()
        self.set_status(200)


class PersonHandler(SurroundHandler):
    def _person_uuid(self, person_id):
        try:
            return uuid.UUID(person_id)
        except ValueError as e:
            raise SurroundWebError(reason=str(e), status_code=400)

    def get(self, person_id):
        result = POSTGRES_CLIENT.find_person_by_id(self._person_uuid(person_id))
        self.write(json_encode(result.__dict__))

    def delete(self, person_id):
        POSTGRES_CLIENT.delete_person_by_id(self._person_uuid(person_id))
        self.set_status(200)


class FaceCollectionHandler(SurroundHandler):
    def get(self):
        faces = POSTGRES_CLIENT.find_all_faces()
        self.write(json_encode([face.serializable() for face in faces]))

    def delete(self):
        POSTGRES_CLIENT.delete_all_faces()
        self.set_status(200)


class FaceHandler(SurroundHandler):
    def get(self, person_id, face_id=None):
        faces = []
        if face_id is not None:
            face = POSTGRES_CLIENT.find_person_face(person_id, face_id)
            self.write(json_encode(face.serializable()))
        else:
            faces = POSTGRES_CLIENT.find_person_faces(person_id)
            self.write(json_encode([face.serializable() for face in faces]))

    def post(self, person_id):
        file_info = self.request.files["file"][0]
        filename = file_info["filename"]
        extension = os.path.splitext(filename[1])

        data = PipelineData(image_filename=filename, image_bytes=file_info["body"])
        SURROUND.process(data)

        if data.error is not None:
            raise SurroundWebError(reason=str(data.error["error"]), status_code=400)
        if len(data.face_encodings) == 0:
            raise SurroundWebError(reason="No face found", status_code=400)
        if len(data.face_encodings) > 1:
            raise SurroundWebError(reason="More than one face in registration photo", status_code=400)

        face = Face(
            id=None,
            person_id=person_id,
            encoding=data.face_encodings[0],
            photo_md5=hashlib.md5(file_info["body"]).hexdigest(),
            photo_filename=filename,
            box_x1=data.face_filtered_boxes[0][0],
            box_x2=data.face_filtered_boxes[0][2],
            box_y1=data.face_filtered_boxes[0][1],
            box_y2=data.face_filtered_boxes[0][3],
            encoder_version="v1",
            encoder_batch_id=uuid.uuid4()
        )

        try:
            result = POSTGRES_CLIENT.create_face_for_person(person_id, face)
        except psycopg2.IntegrityError:
            raise SurroundWebError(reason="Person already has an identical face encoding", status_code=400)

        self.set_status(201)
        self.write(json_encode(result.serializable()))

    def delete(self, person_id, face_id=None):
        if face_id is not None:
            POSTGRES_CLIENT.delete_face_for_person(person_id, face_id)
        else:
            POSTGRES_CLIENT.delete_faces_for_person(person_id)


class SearchHandler(SurroundHandler):
    def prepare(self):
        super().prepare()
        nearest = self.get_query_argument("nearest", None)
        if nearest is not None:
            try:
                nearest = int(nearest)
                if nearest <= 0:
                    raise ValueError
            except ValueError:
                raise SurroundWebError(reason="'nearest' must be a positive integer", status_code=400)

        self.nearest = nearest


class PhotoSearchHandler(SearchHandler):
    def post(self):
        if self.nearest is None:
            self.write("Search for a person using a photo")
        else:
            self.write("Search for the nearest {} persons using a photo".format(self.nearest))


class EncodingSearchHandler(SearchHandler):
    def post(self):
        if self.nearest is None:
            self.write("Search for a person using an encoding")
        else:
            self.write("Search for the nearest {} persons using an encoding".format(self.nearest))


class AdHocEncodingHandler(SurroundHandler):
    def post(self):
        # @TODO: Photo upload --> Surround pipeline
        face = Face(
            id=None,
            person_id=None,
            encoding=Face.dummy_encoding(),
            photo_md5=hashlib.md5(str(uuid.uuid4()).encode("utf-8")).hexdigest(),
            photo_filename="dummy.jpg",
            box_x1=1,
            box_x2=2,
            box_y1=3,
            box_y2=4,
            encoder_version="v1",
            encoder_batch_id=uuid.uuid4()
        )

        self.write(json_encode(face.serializable()))
