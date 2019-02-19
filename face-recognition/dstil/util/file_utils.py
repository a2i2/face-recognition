import argparse
import json
import hashlib
import os


def read_json_file(path):
    """
    Reads the given JSON file into a dictionary.
    """
    return json.loads(open(path, "r").read())


def md5(file_path):
    """
    Returns the MD5 hex digest of the given file.
    """
    with open(file_path, "rb") as f:
        return md5_from_bytes(f.read())


def md5_from_bytes(in_bytes):
    """
    Returns the MD5 hex digest of the given bytes.
    """
    hasher = hashlib.md5()
    hasher.update(in_bytes)
    return hasher.hexdigest()


class ReadableDir(argparse.Action):
    """
    Argparse action for ensuring a directory argument exists and is readable.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))
