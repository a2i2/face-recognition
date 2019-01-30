# utils.py
#
# Contains utility functions used by multiple modules.

import logging
import os
import re
import shutil
import yaml
import stringcase
from dstil.util import file_utils

# Global configuration settings
global_config = dict()

LOGGER = logging.getLogger(__name__)


def matches_expected_naming_format(image_name):
    """
    Returns True if the given image name matches the expected naming format,
    which is an N-digit number, possibly followed by an underscore and another N-digit number.
    e.g. 12345, 12345_67, etc.
    """
    return re.compile("^(\d+)(_\d+)?$").match(image_name)


def get_sorted_path(image_name):
    """
    Given an image name (no extension), returns the name of the directory in which that file would be found
    in the 'sorted' directory structure. The expected naming format for filenames is one or more digits,
    possibly followed by an underscore and more digits.
    """
    target_dir_name = "unsorted"

    if matches_expected_naming_format(image_name):
        person_id = image_name.split("_")[0]
        num_digits = len(person_id)

        if num_digits <= 3:
            # 0-999 go in 0000/
            target_dir_name = "0000"
        else:
            # The rest go in a bucket derived from all but the last three digits, e.g. 12345.jpg --> 12000/12345.jpg
            target_dir_name = person_id[:-3] + "000"

    return target_dir_name


def copy_file(image_path, destination_base_path, input_base_path = None):
    """
    Copies the given image file to the specified destination according to the naming rules in
    get_sorted_path().

    In the event of a path clash (which happens when two source files in different subfolders have the same name),
    we compare the MD5 of each file. Ideally, the MD5 hashes match, and we simply skip the file. If the contents are different,
    we instead copy the image to the `duplicates` folder with its original path intact.

    If specified, input_base_path will be stripped from the output path. This makes it possible to ignore leading paths that
    only exist locally. For instance, for an image path of '/home/documents/dataset/12345.jpg', setting input_base_path to
    '/home/documents/dataset' will ensure that the destination is '/destination/12345.jpg', instead of
    '/destination/home/documents/dataset/12345.jpg'.
    """
    # Extract components from image path.
    #   source_dir_path             ==> /original/path/to
    #   image_filename              ==> 1234567_89.jpg
    #   image_name                  ==> 1234567_89
    source_dir_path, image_filename = os.path.split(image_path)
    image_name = os.path.splitext(image_filename)[0]

    # Join up final paths for copy operation.
    #   target_dir_name             ==> 1234000
    #   target_dir_path             ==> /new/path/to/1234000
    #   target_file_path            ==> /new/path/to/1234000/1234567_89.jpg
    target_dir_name = get_sorted_path(image_name)
    target_dir_path = os.path.join(destination_base_path, target_dir_name)
    target_file_path = os.path.join(target_dir_path, image_filename)

    # Copy image file if it doesn't already exist.
    if not os.path.exists(target_file_path):
        LOGGER.info("Copying {} --> {}".format(image_path, target_file_path))
        os.makedirs(target_dir_path, exist_ok=True)
        shutil.copy(image_path, target_file_path)
    else:
        # If image file already exists, make sure the contents are identical before skipping.
        source_md5 = file_utils.md5(image_path)
        target_md5 = file_utils.md5(target_file_path)
        if source_md5 != target_md5:
            # If contents are different, change the target path to /duplicates/path/to/original/image.jpg
            old_target_file_path = target_file_path
            target_dir_path = os.path.join(destination_base_path, "duplicates", source_dir_path.replace(input_base_path, "").strip(os.sep) if input_base_path is not None else source_dir_path.strip(os.sep))
            target_file_path = os.path.join(target_dir_path, image_filename)
            LOGGER.warning("Target file {} exists but has different contents! Copying to {}".format(old_target_file_path, target_file_path))
            os.makedirs(target_dir_path, exist_ok=True)
            shutil.copy(image_path, target_file_path)
        else:
            LOGGER.info("Identical target file {} already exists; skipping".format(target_file_path))


def get_config():
    """
    Loads global configuration settings from `../../../resources/config/config.yml
    """
    global global_config
    if global_config == dict():
        config_file_path = os.path.realpath(
            os.path.join(os.path.realpath(__file__), "../", "../", "../", "resources", "config", "config.yml"))
        with open(config_file_path) as f:
            global_config = yaml.load(f.read())
        LOGGER.info("Loading configuration file: %s", config_file_path)

        # Override config items with the value of any environment variables that are set.
        # @TODO: This only works for flat config files and not nested keys.
        for key, value in global_config.items():
            env_var_equivalent = stringcase.constcase(key)
            env_var_value = os.environ.get(env_var_equivalent)
            if env_var_value is not None:
                LOGGER.info("Environment variable {} is set; overriding {} with value {} (default value is {})".format(env_var_equivalent, key, str(env_var_value), str(value)))
                global_config[key] = type(value)(env_var_value)

    return global_config


class ErrorCode:

    @staticmethod
    def UNKNOWN_ERROR(error_name, error_message, file, line):
        return 0, "UNKNOWN_ERROR", "{},{},{}:{}".format(error_name, error_message, file, line)

    FACE_NOT_FOUND = (1, "FACE_NOT_FOUND", "A face encoding could not be found in image")
    INVALID_FILE = (2, "INVALID_FILE", "Invalid file that can not be opened as an image")
    NO_MODEL_ENCODING = (3, "NO_MODEL_ENCODING", "No model loaded for extracting face encodings")
    NO_MODEL_FACE_DETECTOR = (4, "NO_MODEL_FACE_DETECTOR", "No model loaded for detecting faces")
    NO_MODEL_ROTATION = (5, "NO_MODEL_ROTATION", "No model loaded for detection image rotation")

def error_values():
    values = ErrorCode.__dict__
    return [values[k] for k in values if not k[0] == "_"]


class WarningCode:

    CREATED_ENCODING = (0, "CREATED_ENCODING", "Encoding had to be created for the image")
    INVALID_EXIF_DATA = (1, "INVALID_EXIF_DATA", "EXIF data could not be read correctly")
    FACE_TOO_SMALL = (2, "FACE_TOO_SMALL", "Face in image is too small for accurate matching")
    IMAGE_TOO_DARK = (3, "IMAGE_TOO_DARK", "Image is too dark for optimal face matching")
    FACE_TOO_BLURRY = (4, "FACE_TOO_BLURRY", "Face is too blurry for optimal face matching")
    IMAGE_ROTATED = (5, "IMAGE_ROTATED", "Image had to be rotated to find a face")
    IMAGE_ROTATION_UNDEFINED = (6, "IMAGE_ROTATION_UNDEFINED", "Image rotation value below the rotation threshold")
