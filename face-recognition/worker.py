import os
import surround
import logging
from .stages import face_recognition_pipeline, FaceRecognitionPipelineData
from celery import Celery, Task
from celery.signals import worker_process_init
from surround import Config
import uuid

logging.basicConfig(level=logging.ERROR)
app = Celery("face-recognition", broker="pyamqp://guest@localhost", backend="redis://localhost")
pipeline = None
config = None


@worker_process_init.connect()
def init_worker_process(**kwargs):
    global pipeline
    global config

    config = Config()
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
    config.read_config_files([config_path])

    pipeline = face_recognition_pipeline()
    surround.surround.LOGGER.setLevel(logging.ERROR)
    pipeline.set_config(config)
    pipeline.init_stages()


@app.task(time_limit=5)
def encode(path):
    data = FaceRecognitionPipelineData(path)
    pipeline.process(data)
    return data.output_data
