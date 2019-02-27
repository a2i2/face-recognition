# A²I² Face Recognition Service

> An example project for Surround using face recognition models.

This project demonstrates how Surround can be used to build a facial recognition pipeline.

## Usage

For usage instructions, run `python3 -m facerecognition -h`.

### Summary

The pipeline can be run in three different modes: `server`, `batch`, and `worker`:
* `server` mode:
  - Usage: `python3 -m facerecognition server [-w]`
  - Description: Runs an HTTP server that provides REST endpoints for person/face registration and recognition. Additionally, the `-w`/`--webcam` flag can be used to run a TCP server alongside the HTTP server that will provide frames and face detection from a local webcam. NOTE: This currently only works on Linux. See the [Known Issues](#known-issues) section.
* `batch` mode:
  - Usage: `python3 -m facerecognition batch`
  - Description: Processes a directory of image files and produce an encoding for each one.
* `worker` mode:
  - Usage: `celery -A facerecognition.worker worker`
  - Description: Run a Celery worker that will listen to the configured broker for face encoding jobs.

For detailed usage, supply `-h` to each subcommand.

### Server mode

#### Endpoints

To view a list of endpoints, visit the `/info` endpoint. A summarised list is as follows:

  * `POST /persons`: Create a person
  * `GET /persons`: Get all persons
  * `GET /persons?name=name`: Search for a person by name
  * `GET /persons/:id`: Get a single person by ID
  * `GET /persons/:id/faces`: Get all face encodings for a person
  * `POST /persons/:id/faces`: Add a face to a person
  * `GET /persons/:id/faces/:id`: Get a single face for a person by ID
  * `DELETE /persons/:id/faces/:id`: Delete a face for a person
  * `DELETE /persons/:id/faces`: Delete all faces for a person
  * `DELETE /persons`: Delete all persons
  * `DELETE /persons/:id`: Delete a person by ID
  * `POST /persons/photo-search`: Search for a person using a photo
  * `POST /persons/photo-search?nearest=N`: Get a list of the nearest people (in order of confidence) using a photo, up to a maximum of N people
  * `POST /persons/encoding-search`: Search for a person using an encoding
  * `POST /persons/encoding-search?nearest=N`: Get a list of the nearest people (in order of confidence) using a photo, up to a maximum of N people
  * `GET /faces`: Get all face encodings
  * `DELETE /faces`: Delete all face encodings
  * `POST /encode`: Perform a once-off encoding (don't save anything)"

#### Postman collection

An easy way to test the endpoints is to import the [Postman](https://www.getpostman.com/) collection and environment from the `postman` folder.

![Postman Screenshot](postman_screenshot.png)

### Batch mode

Batch mode can be used to encode a directory of images and produce an encoding for each one. To test, run the following:

```
python3 -m facerecognition -i data/input -o data/output -c facerecognition/config.yaml
```

Then, check the `data/output` directory for encoding output.

### Worker mode

Worker mode can be used to encode large volumes of images. This mode requires a RabbitMQ and Redis server for job/result management. The easiest way to test this is to run via docker-compose:

```
docker-compose -f docker-compose-distributed.yml up
```

This will run a worker alongside all required backing services, and will share the `data/input` directory into the Docker container to the target path `/var/lib/face-recognition/input`.

You can visit the Flower dashboard at `localhost:5555` to view worker/job status, and send jobs via cURL:

```
curl -X POST -d '{"args":["/var/lib/face-recognition/input/beyonce.PNG"]}' http://localhost:5555/api/task/async-apply/face-recognition.worker.encode
```

You can also interact with the workers from the Python repl as follows:

```
>>> from facerecognition.worker import encode
>>> encode.delay("/var/lib/face-recognition/input/beyonce.PNG")
<AsyncResult: bbe00661-d417-4596-b122-30e23df8beff>
>>>
```

## Docker image

This project is available as a Docker image, which can be run via the following command (As above, supply `server` or `batch` to choose the operating mode):

```
docker run dstilab/face-recognition
```

Running in `server` mode requires a PostgreSQL server to store registered people/faces. The easiest way to do this is to run via docker-compose:

```
docker-compose up
```

## Known issues

1. The webcam feed only works on Linux because [extra steps](https://stackoverflow.com/questions/41023827/accessing-usb-webcam-hosted-on-os-x-from-a-docker-container) are required to share a webcam with Docker on OSX/Windows hosts.
2. `worker` mode currently does nothing with the results; it is there mostly for demonstration purposes.
