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


def camel_to_snake(camel_str):
    """
    Converts the given snake_case string to camelCase.
    :param camel_str: String in camelCase format
    :return snake_case representation of the input string
    """
    camel_pat = re.compile(r'([A-Z])')
    return camel_pat.sub(lambda x: '_' + x.group(1).lower(), camel_str)


def snake_to_camel(snake_str):
    """
    Converts the given camelCase string to snake_case.
    :param snake_str: String in snake_case format
    :return camelCase representation of the input string
    """
    under_pat = re.compile(r'_([a-z])')
    return under_pat.sub(lambda x: x.group(1).upper(), snake_str)


def change_dict_naming_convention(in_dict, convert_function):
    """
    Converts a nested dictionary from one casing convention to another.
    :param in_dict: Dictionary to be converted
    :param convert_function: Function to convert key casing convention
    :return Dictionary with the new keys
    """
    if isinstance(in_dict, dict):
        result = dict()
        for key, val in in_dict.items():
            result[convert_function(key)] = change_dict_naming_convention(val, convert_function)
    elif isinstance(in_dict, list):
        result = list()
        for item in in_dict:
            result.append(change_dict_naming_convention(item, convert_function))
    else:
        result = in_dict

    return result


def serialize(val):
    """
    Serialises the given val to a JSON-compatible string, handling edge cases
    such as datetimes and enums.
    """
    result = None

    if isinstance(val, datetime.datetime):
        # Appending Z is only valid when timezone is GMT/UTC (+00:00). RFC3339 library should handle this
        result = rfc3339.rfc3339(val)
    elif isinstance(val, enum.Enum):
        result = val.value
    elif isinstance(val, Exception):
        result = dict(error=val.__class__.__name__, args=val.args)
    else:
        result = str(val)

    return result


def to_json(obj, pretty=False):
    """
    Serialises the given *obj* to a JSON string. If *pretty* is True, the JSON is
    printed in human-friendly format.
    """
    assert not isinstance(obj, list)
    if isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = obj.__dict__

    if pretty:
        return json.dumps(
            change_dict_naming_convention(obj_dict, snake_to_camel),
            default=serialize,
            sort_keys=True,
            indent=4
        )
