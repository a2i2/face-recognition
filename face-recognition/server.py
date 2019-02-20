import datetime
import traceback
import json
import tornado.web
from tornado.escape import json_encode, json_decode


START_TIME = datetime.datetime.now()


class SurroundWebApplication(tornado.web.Application):
    def __init__(self, **kwargs):
        kwargs["handlers"] = [
            (r"/", MainHandler),
            (r"/info", InfoHandler),
            (r"/persons/(?P<person_id>\w+)", PersonHandler),
            (r"/persons", PersonCollectionHandler),
            (r"/persons/photo-search", PhotoSearchHandler),
            (r"/persons/encoding-search", EncodingSearchHandler),
            (r"/persons/(?P<person_id>\w+)/faces/(?P<face_id>\w+)", FaceHandler),
            (r"/persons/(?P<person_id>\w+)/faces", FaceHandler),
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
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            if len(self.request.body) > 0:
                self.json_body = json_decode(self.request.body.decode("utf-8"))
            else:
                self.json_body = dict()
        else:
            self.json_body = None

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
        if name is not None:
            self.write("Get persons named " + name)
        else:
            self.write("Get all persons")

    def post(self):
        self.write("Create new person")

    def delete(self):
        self.write("Delete all persons")


class PersonHandler(SurroundHandler):
    def get(self, person_id):
        self.write("Get person {}".format(person_id))

    def delete(self, person_id):
        self.write("Delete person " + person_id)


class FaceCollectionHandler(SurroundHandler):
    def get(self):
        self.write("Get all faces")

    def delete(self, person_id=None, face_id=None):
        self.write("Delete all faces")


class FaceHandler(SurroundHandler):
    def get(self, person_id, face_id=None):
        if face_id is not None:
            self.write("Get face {} for person {}".format(face_id, person_id))
        else:
            self.write("Get all faces for person {}".format(person_id))

    def post(self, person_id):
        self.write("Create new face for person {}".format(person_id))

    def delete(self, person_id, face_id=None):
        if face_id is not None:
            self.write("Delete face {} for person {}".format(face_id, person_id))
        else:
            self.write("Delete all faces for person {}".format(person_id))


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
        self.write("Encode a photo")
