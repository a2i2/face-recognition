import json
import numpy


class Person:
    """
    Represents a 'person' with a name that can be used
    to map between the face recognition server and
    a domain-specific application.
    """
    def __init__(self, id, name):
        self.id = id
        self.name = name

class Face:
    """
    Represents a 'face' belonging to a given person.
    """
    def __init__(self, **kwargs):
        try:
            self.id = kwargs.get("id", None)
            self.person_id = kwargs.get("person_id")
            self.encoding = kwargs.get("encoding")
            self.photo_md5 = kwargs.get("photo_md5")
            self.photo_filename = kwargs.get("photo_filename")
            self.box_x1 = kwargs.get("box_x1")
            self.box_x2 = kwargs.get("box_x2")
            self.box_y1 = kwargs.get("box_y1")
            self.box_y2 = kwargs.get("box_y2")
            self.encoder_version = kwargs.get("encoder_version")
            self.encoder_batch_id = kwargs.get("encoder_batch_id")
        except KeyError as e:
            raise FaceRecognitionError(reason="Missing field: " + str(e), status_code=400)

    @staticmethod
    def dummy_encoding():
        return [float(num) for num in numpy.random.uniform(-0.2, 0.2, size=(1, 128))[0]]

    def serializable(self):
        entries = self.__dict__
        if entries["id"] is not None:
            entries["id"] = str(entries["id"])
        if entries["person_id"] is not None:
            entries["person_id"] = str(entries["person_id"])
        if entries["encoder_batch_id"] is not None:
            entries["encoder_batch_id"] = str(entries["encoder_batch_id"])

        entries["encoding"] = [float(value) for value in entries["encoding"]]
        return entries
