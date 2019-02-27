import logging
import argparse
import os
import sys
import json
from surround import Surround, Config
from .stages import face_recognition_pipeline, FaceRecognitionPipelineData
from .server import FaceRecognitionWebApplication
from .webcam_tcp_server import WebcamServer
from tornado.ioloop import IOLoop
from .utils import is_valid_dir, is_valid_file, iglob_recursive


# Set up default logging config.
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def argument_parser():
    """
    Sets up the parser for command line arguments.
    We use a subparser for each 'mode' to divide arguments into mutually inclusive groups.
    """
    parser = argparse.ArgumentParser(description="A2I2 Face Recognition")
    subparsers = parser.add_subparsers(
        title="Operating modes",
        description="A2I2 Face Recognition can run in 'server' mode or 'batch' mode.",
        dest="mode")
    server = subparsers.add_parser("server", help="Run an HTTP server with REST endpoints for performing face registration/recognition")
    server.add_argument("-w", "--webcam", help="Run a TCP server alongside the HTTP server that serves a video stream from a local webcam", required=False, action="store_true")
    batch = subparsers.add_parser("batch", help="Process a directory of image files and produce an encoding for each one")
    batch.add_argument("-o", "--output-dir", required=True, help="Output directory",
                                         type=lambda x: is_valid_dir(batch, x))
    batch.add_argument("-i", "--input-dir", required=True, help="Input directory",
                                         type=lambda x: is_valid_dir(batch, x))
    batch.add_argument("-c", "--config-file", required=True, help="Path to config file",
                                         type=lambda x: is_valid_file(batch, x))

    return parser


def process_image_dir(input_dir, output_dir, config_path):
    """
    Processes all image files in the given input_dir and outputs face encodings
    (or an error) for each one in the given output_dir, using the config file located
    at config_path.
    """
    # Load config from the specified file.
    config = Config()
    config.read_config_files([config_path])

    # Load and initialise the face encoding pipeline.
    pipeline = face_recognition_pipeline()
    pipeline.set_config(config)
    pipeline.init_stages()

    # Process each image in input_dir.
    for filename in iglob_recursive(input_dir, "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        LOGGER.info("Processing {}...".format(filename))

        # Run the current filename through the pipeline.
        data = FaceRecognitionPipelineData(filename)
        pipeline.process(data)

        if data.error:
            # Check and handle any errors.
            LOGGER.error(str(data.error))
            output_filename = "{}.error.json".format(os.path.basename(filename))
            with open(os.path.abspath(os.path.join(output_dir, output_filename)), "w") as output_file:
                output_file.write(json.dumps(data.error))
        else:
            # Write the output to file.
            output_filename = "{}.encoding.json".format(os.path.basename(filename))
            with open(os.path.abspath(os.path.join(output_dir, output_filename)), "w") as output_file:
                output = dict(output=data.output_data, warnings=data.warnings)
                output_file.write(json.dumps(output))

            # Log any warnings.
            for warning in data.warnings:
                LOGGER.warning(str(warning))


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argument_parser()
    args = parser.parse_args()

    # Run pipeline in 'server' or 'batch' mode.
    if args.mode == "server":
        http_server = FaceRecognitionWebApplication(debug=False)
        http_server.listen(8888)
        logging.info("HTTP server listening on 8888")

        if args.webcam:
            webcam_server = WebcamServer()
            webcam_server.listen(8889)
            if not webcam_server.vs.stopped:
                logging.info("Webcam TCP server listening on 8889")

        IOLoop.instance().start()
        logging.info("Server has shut down.")
    elif args.mode == "batch":
        process_image_dir(args.input_dir, args.output_dir, args.config_file)
    else:
        parser.print_help(sys.stderr)
