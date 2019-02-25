CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE person (
    id    UUID          PRIMARY KEY  DEFAULT gen_random_uuid(),
    name  VARCHAR(255)  NOT NULL
);

CREATE TABLE face (
    id               UUID         PRIMARY KEY  DEFAULT gen_random_uuid(),
    person_id        UUID         NOT NULL     REFERENCES person (id),
    encoding         BYTEA        NOT NULL,
    photo_md5        VARCHAR(32)  NOT NULL,
    photo_filename   VARCHAR(255) NOT NULL,
    box_x1           INT          NOT NULL,
    box_x2           INT          NOT NULL,
    box_y1           INT          NOT NULL,
    box_y2           INT          NOT NULL,
    encoder_version  VARCHAR(64)  NOT NULL,
    encoder_batch_id UUID         NOT NULL,

    UNIQUE(encoding, photo_md5)
);

CREATE INDEX person_name_idx           ON person(name);
CREATE INDEX face_person_id_idx        ON face(person_id);
CREATE INDEX face_photo_md5_idx        ON face(photo_md5);
CREATE INDEX face_photo_filename_idx   ON face(photo_filename);
CREATE INDEX face_encoder_version_idx  ON face(encoder_version);
CREATE INDEX face_encoder_batch_id_idx ON face(encoder_batch_id);
