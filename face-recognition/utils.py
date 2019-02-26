import os
import numpy
import itertools
import glob
import datetime
import enum
import json
import rfc3339
from dateutil import parser


def is_valid_dir(arg_parser, arg):
    """
    Parses the given argument and raises an error if it is not
    a valid directory on the local filesystem.
    """
    if not os.path.isdir(arg):
        arg_parser.error("Invalid directory {}".format(arg))
    else:
        return arg


def is_valid_file(arg_parser, arg):
    """
    Parses the given argument and raises an error if it is not
    a valid file on the local filesystem.
    """
    if not os.path.isfile(arg):
        arg_parser.error("Invalid file {}".format(arg))
    else:
        return arg


def iglob_recursive(start_dir, *patterns):
    """
    Returns an iterator that yields globbed file paths anywhere
    under the given start_dir, recursively.
    """
    return itertools.chain.from_iterable(
        glob.iglob("{}/**/{}".format(start_dir, pattern), recursive=True)
        for pattern in patterns)


def distance(encoding1, encoding2):
    """
    Calculate the euclidean distance between two face encodings.
    An encoding is represented by a 1x128 vector of doubles.
    """
    encoding1 = numpy.array(encoding1)
    encoding2 = numpy.array(encoding2)

    return numpy.linalg.norm(encoding2 - encoding1)
