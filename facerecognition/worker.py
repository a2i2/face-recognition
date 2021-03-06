import os
import surround
import logging
from .stages import face_recognition_pipeline, FaceRecognitionPipelineData
from celery import Celery, Task
from celery.signals import worker_process_init
from surround import Config


# Set up logging.
logging.basicConfig(level=logging.ERROR)

# Load config from file.
config = Config()
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
config.read_config_files([config_path])

# Create Celery app and configure its queue broker and results backend.
app = Celery(
    "face-recognition",
    broker=config["celery"]["broker"],
    backend=config["celery"]["backend"])

# TensorFlow is not fork-safe, so we need to initialise the
# Surround pipeline in @worker_process_init.connect() instead
# of doing it here.
pipeline = None

@worker_process_init.connect()
def init_worker_process(**kwargs):
    """
    Called in all pool child processes when they start.
    Initialises the Surround pipeline for the worker.
    """
    global pipeline
    global config

    pipeline = face_recognition_pipeline()
    surround.surround.LOGGER.setLevel(logging.ERROR)
    pipeline.set_config(config)
    pipeline.init_stages()


@app.task(time_limit=60)
def encode(path):
    """
    Worker task that takes a file path and returns
    a face encoding.
    """
    data = FaceRecognitionPipelineData(path)
    pipeline.process(data)
    return data.output_data
