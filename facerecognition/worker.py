import os
import surround
import logging
from .stages import face_recognition_pipeline, FaceRecognitionPipelineData
from celery import Celery, Task
from celery.signals import worker_process_init
from surround import Config

logging.basicConfig(level=logging.ERROR)

config = Config()
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
config.read_config_files([config_path])

app = Celery(
    "face-recognition",
    broker=config["celery"]["broker"],
    backend=config["celery"]["backend"])

pipeline = None


@worker_process_init.connect()
def init_worker_process(**kwargs):
    global pipeline
    global config

    pipeline = face_recognition_pipeline()
    surround.surround.LOGGER.setLevel(logging.ERROR)
    pipeline.set_config(config)
    pipeline.init_stages()


@app.task(time_limit=60)
def encode(path):
    data = FaceRecognitionPipelineData(path)
    pipeline.process(data)
    return data.output_data
