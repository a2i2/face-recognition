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
from tornado.escape import json_decode
from .postgres_client import PostgresClient
from .entities import Person, Face
from .stages import face_recognition_pipeline, FaceRecognitionPipelineData
from surround import Surround, Config
from .utils import distance, to_json

# Get time for uptime calculation.
START_TIME = datetime.datetime.now()

# Load config file.
CONFIG = Config()
CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
CONFIG.read_config_files([CONFIG_PATH])

# Create PostgreSQL client.
POSTGRES_CLIENT = PostgresClient(CONFIG["postgres"]["db"], CONFIG["postgres"]["user"], CONFIG["postgres"]["host"], CONFIG["postgres"]["port"], CONFIG["postgres"]["password"])

# Create face recognition pipeline.
FACE_RECOGNITION_PIPELINE = face_recognition_pipeline()
FACE_RECOGNITION_PIPELINE.set_config(CONFIG)
FACE_RECOGNITION_PIPELINE.init_stages()


class FaceRecognitionWebApplication(tornado.web.Application):
    def __init__(self, **kwargs):
        kwargs["handlers"] = [
            (r"/", HomeHandler),
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
        """
        Handles any errors that occur, returning an appropriate
        JSON response and HTTP status code.
        """
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


class HomeHandler(FaceRecognitionWebHandler):
    """
    Top-level handler for the root route.
    """
    def get(self):
        self.write(dict(
            app="A2I2 Face Recognition",
            version="0.1",  # @TODO: Link up to actual version
            uptime=str(datetime.datetime.now() - START_TIME)
        ))


class InfoHandler(FaceRecognitionWebHandler):
    """
    Handler for the /info route. Displays a list of supported endpoints
    with accompanying descriptions.
    """
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
            dict(request="DELETE /persons/:id",                     description="Delete a person by ID"),
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
    """
    Hander for endpoints that deal with the overall Person collection.
    """
    def get(self):
        """Finds all people, or a subset thereof according to the given search criteria."""
        name = self.get_query_argument("name", None)
        people = []
        if name is not None:
            people = POSTGRES_CLIENT.find_persons_by_name(name)
        else:
            people = POSTGRES_CLIENT.find_all_persons()

        self.write(dict(persons=[person.__dict__ for person in people]))

    def post(self):
        """Creates a new person."""
        try:
            person = Person(id=None, name=self.json_body["name"])
        except KeyError as e:
            raise FaceRecognitionError(reason="Missing field: " + str(e), status_code=400)

        result = POSTGRES_CLIENT.create_person(person)
        self.set_status(201)
        self.write(result.__dict__)

    def delete(self):
        """Deletes all people."""
        POSTGRES_CLIENT.delete_all_persons()
        self.set_status(200)


class PersonHandler(FaceRecognitionWebHandler):
    """
    Hander for endpoints that deal with a specific Person.
    """
    def _person_uuid(self, person_id):
        """Parses a UUID string into an actual UUID object."""
        try:
            return uuid.UUID(person_id)
        except ValueError as e:
            raise FaceRecognitionError(reason=str(e), status_code=400)

    def get(self, person_id):
        """Fetches a person with the given ID."""
        result = POSTGRES_CLIENT.find_person_by_id(self._person_uuid(person_id))
        self.write(result.__dict__)

    def delete(self, person_id):
        """Deletes a person with the given ID."""
        POSTGRES_CLIENT.delete_person_by_id(self._person_uuid(person_id))
        self.set_status(200)


class FaceCollectionHandler(FaceRecognitionWebHandler):
    """
    Hander for endpoints that deal with the overall Face collection.
    """
    def get(self):
        """Fetches all faces."""
        faces = POSTGRES_CLIENT.find_all_faces()
        self.write(dict(faces=[face.serializable() for face in faces]))

    def delete(self):
        """Deletes all faces."""
        POSTGRES_CLIENT.delete_all_faces()
        self.set_status(200)


class PipelineHandler(FaceRecognitionWebHandler):
    """
    Base class for handlers that run the face recognition pipeline.
    """
    def _pipeline_data_from_photo(self):
        """Returns a FaceRecognitionPipelineData object loaded from an uploaded file."""
        try:
            file_info = self.request.files["file"][0]
        except KeyError:
            raise FaceRecognitionError(reason="No file uploaded via 'file' form-data field")
        return FaceRecognitionPipelineData(image_filename=file_info["filename"], image_bytes=file_info["body"])

    def _run_pipeline(self, data, single_face=True):
        """Runs the face recognition pipeline on uploaded image data."""
        FACE_RECOGNITION_PIPELINE.process(data)

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
    """
    Handler for endpoints that deal with faces for a given person.
    """
    def get(self, person_id, face_id=None):
        """Finds one or all faces for the given person."""
        faces = []
        if face_id is not None:
            face = POSTGRES_CLIENT.find_person_face(person_id, face_id)
            self.write(face.serializable())
        else:
            faces = POSTGRES_CLIENT.find_person_faces(person_id)
            self.write(dict(faces=[face.serializable() for face in faces]))

    def post(self, person_id):
        """Uploads and processes a new face for the given person."""
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
        """Deletes one or all faces for the given person."""
        if face_id is not None:
            POSTGRES_CLIENT.delete_face_for_person(person_id, face_id)
        else:
            POSTGRES_CLIENT.delete_faces_for_person(person_id)


class SearchHandler(PipelineHandler):
    """
    Base class for handlers that will search for similar faces to a given
    photo or encoding.
    """
    def prepare(self):
        """Processes query parameters that control how many results are returned."""
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
        """Searches for people with faces similar to the given encoding."""
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
    """
    Handler for searches that use a photo as input.
    """
    def post(self):
        """Runs an uploaded image through the pipeline and returns the N closest people."""
        data = self._pipeline_data_from_photo()
        face = self._run_pipeline(data, single_face=True)

        if self.nearest is None:
            closest = self._closest_people(face.encoding, limit=1)
            self.write(closest["0"])
        else:
            closest = self._closest_people(face.encoding, limit=self.nearest)
            self.write(closest)


class EncodingSearchHandler(SearchHandler):
    """
    Handler for searches that use an encoding as input.
    """
    def post(self):
        """Returns the N closest people with faces similar to the provided encoding."""
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
    """
    Handler for producing face encodings from photos on an ad-hoc basis.
    """
    def post(self):
        data = self._pipeline_data_from_photo()
        face = self._run_pipeline(data, single_face=True)

        self.write(face.serializable())
