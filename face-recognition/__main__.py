import logging
import argparse
import os
import json
from surround import Surround, Config
from .stages import *
from .server import FaceRecognitionWebApplication
from .webcam_tcp_server import WebcamServer
from tornado.ioloop import IOLoop

# Data Science projects should use logging over print statements:
#
#    import logging
#    logging.info("print to output")
#
# This is so that in a production environment the output is written to
# log files for debugging rather than to standard out where the output
# is lost.
# The command below configures the default logger.
logging.basicConfig(level=logging.INFO)

# Validation functions for the command line parser below
def is_valid_dir(arg_parser, arg):
    if not os.path.isdir(arg):
        arg_parser.error("Invalid directory %s" % arg)
    else:
        return arg

def is_valid_file(arg_parser, arg):
    if not os.path.isfile(arg):
        arg_parser.error("Invalid file %s" % arg)
    else:
        return arg

# Set up the parser for command line arguments
parser = argparse.ArgumentParser(description="The Surround Command Line Interface")
# parser.add_argument('-o', '--output-dir', required=True, help="Output directory",
#                                      type=lambda x: is_valid_dir(parser, x))
#
# parser.add_argument('-i', '--input-dir', required=True, help="Input directory",
#                                      type=lambda x: is_valid_dir(parser, x))
# parser.add_argument('-c', '--config-file', required=True, help="Path to config file",
#                                      type=lambda x: is_valid_file(parser, x))

def load_data(input_dir, output_dir, config_path):

    # All Surround projects have a Surround class which is responsible
    # for configurating and running a list of stages.
    surround = Surround([PhotoExtraction(), DownsampleImage(), RotateImage(), ImageTooDark(), DetectAndAlignFaces(), LargestFace(), FaceTooBlurry(), ExtractEncodingsResNet1()])

    # Config is a dictionary with some utility functions for reading
    # values in from yaml files. Any value read in from a yaml file
    # can be overridden with an environment variable with a SURROUND_
    # prefix and '_' to mark nesting. A yaml file with the following
    # content:
    #
    # output:
    #   mode: True
    #
    # Can be overridden with an environment variable:
    # SURROUND_OUTPUT_MODE
    config = Config()

    # When specifying multiple config files each config file can
    # override a previous config's values.
    config.read_config_files([config_path])

    surround.set_config(config)
    surround.init_stages()

    # Surround operates on an instance of SurroundData. In this case
    # the input data is a string 'data'. See stages.py for more
    # details. In most projects the input for Surround will be read
    # from a file stored in the directory input_dir.
    data = PipelineData(input_dir + "/beyonce.PNG")

    # Start running each stage of Surround by passing the data to
    # each stage in order. The data variable will be updated with the
    # output of each stage so there is no return value.
    surround.process(data)

    # Write the output to file
    with open(os.path.abspath(os.path.join(output_dir, "output.txt")), 'w') as f:
        f.write(json.dumps(data.output_data))

    # Check and handle any errors
    if data.error:
        logging.error("Processing error...")

    # Log all warnings from running the pipeline
    for warn in data.warnings:
        logging.warn(warn)

    # Log the result to screen
    logging.info(data.output_data)

if __name__ == "__main__":
    args = parser.parse_args()
    http_server = FaceRecognitionWebApplication(debug=False)
    http_server.listen(8888)
    webcam_server = WebcamServer()
    webcam_server.listen(8889)

    logging.info("HTTP server listening on 8888")
    IOLoop.instance().start()
    logging.info("Server has shut down.")

    # load_data(args.input_dir, args.output_dir, args.config_file)
