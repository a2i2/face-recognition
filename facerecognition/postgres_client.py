import argparse
import base64
import json
import logging
import psycopg2
import psycopg2.extras
import struct
import time
import uuid
import numpy as np
from .entities import Person, Face

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)

class NotFoundError(Exception):
    """
    Error that can be raised when a query yields no result.
    """
    pass


class PostgresClient:
    """
    Connects to the face recognition database to perform registration and search queries.
    """
    def __init__(self, dbname, user, host, port, password):
        self.dbname = dbname
        self.user = user
        self.host = host
        self.port = port
        self.password = password
        self.postgres = None
        self.retry_time = 5

    def connect(self):
        """Provides a connection to the PostgreSQL database."""
        LOGGER.info("Connecting to db: {}, user: {}, host: {}, port: {}".format(self.dbname, self.user, self.host, self.port))
        self.close()

        while not self.postgres:
            try:
                self.postgres = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, port=self.port, password=self.password, cursor_factory=psycopg2.extras.NamedTupleCursor)
            except Exception as e:
                LOGGER.exception(e)

            if not self.postgres:
                LOGGER.info("Failed to connect to Postgres, trying again in %d second(s)", self.retry_time)
                time.sleep(self.retry_time)

        return self.postgres

    def close(self):
        """Closes the connection to the PostgreSQL database."""
        if self.postgres:
            self.postgres.close()
            self.postgres = None

    def _fetch_person(self, cursor):
        """(internal) Fetches a single Person from the result set."""
        result = cursor.fetchone()
        if result is None:
            raise NotFoundError()

        return Person(id=result.id, name=result.name)

    def _fetch_persons(self, cursor):
        """(internal) Fetches a list of Persons from the result set."""
        results = cursor.fetchall()
        people = []
        for result in results:
            people.append(Person(id=result.id, name=result.name))

        return people

    def _fetch_face(self, cursor):
        """(internal) Fetches a single Face from the result set."""
        result = cursor.fetchone()
        if result is None:
            raise NotFoundError()

        return Face(
            id=result.id,
            person_id=result.person_id,
            encoding=self.base64_decode_double_array(result.encoding),
            photo_md5=result.photo_md5,
            photo_filename=result.photo_filename,
            box_x1=result.box_x1,
            box_x2=result.box_x2,
            box_y1=result.box_y1,
            box_y2=result.box_y2,
            encoder_version=result.encoder_version,
            encoder_batch_id=result.encoder_batch_id
        )

    def _fetch_faces(self, cursor):
        """(internal) Fetches a list of Faces from the result set."""
        results = cursor.fetchall()
        faces = []
        for result in results:
            faces.append(Face(
                id=uuid.UUID(result.id),
                person_id=uuid.UUID(result.person_id),
                encoding=self.base64_decode_double_array(result.encoding),
                photo_md5=result.photo_md5,
                photo_filename=result.photo_filename,
                box_x1=result.box_x1,
                box_x2=result.box_x2,
                box_y1=result.box_y1,
                box_y2=result.box_y2,
                encoder_version=result.encoder_version,
                encoder_batch_id=uuid.UUID(result.encoder_batch_id)
            )
        )

        return faces

    def create_person(self, person):
        assert isinstance(person, Person)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO person (name) VALUES (%s) RETURNING *", (person.name,))
                return self._fetch_person(cursor)

    def find_all_persons(self):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM person")
                return self._fetch_persons(cursor)

    def find_person_by_id(self, id):
        assert isinstance(id, uuid.UUID)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM person WHERE id = %s", (str(id),))
                return self._fetch_person(cursor)

    def find_persons_by_ids(self, id_list):
        str_ids = []
        for id in id_list:
            assert isinstance(id, uuid.UUID)
            str_ids.append(str(id))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM person WHERE id IN (%s)", (",".join(str_ids),))
                return self._fetch_persons(cursor)

    def find_persons_by_name(self, name):
        assert isinstance(name, str)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM person WHERE name = %s", (name,))
                return self._fetch_persons(cursor)

    def delete_person_by_id(self, person_id):
        assert isinstance(person_id, uuid.UUID)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM face WHERE person_id = %s", (str(person_id),))
                cursor.execute("DELETE FROM person WHERE id = %s", (str(person_id),))

    def delete_all_persons(self):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM face")
                cursor.execute("DELETE FROM person")

    def delete_all_faces(self):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM face")

    def find_all_faces(self):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM face")
                return self._fetch_faces(cursor)

    def find_person_face(self, person_id, face_id):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM face WHERE id = %s AND person_id = %s", (face_id, person_id))
                return self._fetch_face(cursor)

    def find_person_faces(self, person_id):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM face WHERE person_id = %s", (person_id,))
                return self._fetch_faces(cursor)

    def create_face_for_person(self, person_id, face):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO face (person_id, encoding, photo_md5, photo_filename, box_x1, box_x2, box_y1, box_y2, encoder_version, encoder_batch_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (str(face.person_id), self.base64_encode_double_array(face.encoding), face.photo_md5, face.photo_filename,
                     face.box_x1, face.box_x2, face.box_y1, face.box_y2, face.encoder_version, str(face.encoder_batch_id)))
                return self._fetch_face(cursor)

    def delete_face_for_person(self, person_id, face_id):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM face WHERE person_id = %s AND id = %s", (person_id, face_id))

    def delete_faces_for_person(self, person_id):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM face WHERE person_id = %s", (person_id,))

    def base64_encode_double_array(self, double_array):
        return base64.b64encode(struct.pack(str(len(double_array)) + 'd', *double_array))

    def base64_decode_double_array(self, base64_string):
        return np.frombuffer(base64.decodestring(base64_string.tobytes()))
