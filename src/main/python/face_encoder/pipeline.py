# pipeline.py
#
# Contains the pipeline for extracting face encodings from a photo. Each stage of the pipeline is represented by an
# implementation of the `Transformation` class. The `process_photo` function is responsible for constructing and
# running the pipeline.
#
# Note: That the pipeline can be configured to dump the output after each stage set by the configuration key
# `enableStageOutputDump`.

import sys
import cv2
import imghdr
import logging
import numpy as np
import os
import struct
import piexif
import traceback
from tensorflow.python.platform import gfile
import tensorflow as tf
from abc import ABC, abstractmethod
from os.path import basename


from .utils import ErrorCode, error_values, get_config, WarningCode, matches_expected_naming_format
from .align import create_mtcnn, detect_face
from datetime import datetime
from dstil.util import file_utils
from surround import Stage, SurroundData, Surround, Config


LOGGER = logging.getLogger(__name__)


class PipelineData(SurroundData):
    """
        Stores the data to be passed between each stage of a pipeline. Note that different stages of the pipeline are
        responsible for setting the attributes.
    """
    image_bytes = None
    image_array = None              # Numpy array of the image stored in BGR format
    image_rgb_array = None          # Image in RGB format
    image_filename = None          # File name of the image
    image_exif = None               # Exif data for the image
    image_grayscale = None          # Grayscale version of the image
    face_bounding_boxes = list()    # List of tuples, (x1, y1, x2, y2) bounding boxes for faces found in the image
                                    # where (x1,y1) is the top left point at (x2, y2) is the bottom right point.
    face_filtered_boxes = list()     # Filtered list of bounding boxes of faces in the image in the form (x1, y1, x2, y2)
    face_encodings = list()          # List of 128 floats representing a face encoding

    def __init__(self, image_filename):
        self.image_filename = image_filename

class Transformation(ABC):
    @abstractmethod
    def transform(self, pipeline_output, pipeline_data):
        """
            A data transformation stage in a pipeline.
            :param pipeline_output: Dictionary object that is the final output for the pipeline
            :param pipeline_data: Stores intermediate data from each stage in the pipeline. Implementations of this
            method should add stage output values to this object and return it as `pipeline_data`.
            :return: Returns the updated pipeline_output object, the updated pipeline_data object, an errors object and a warning dictionary.
            If the error object is not None then the pipeline processing will exit. An error should be one of the
            attributes stored in utils.ErrorCode. The return type is a tuple of the form:
            (pipeline_output, pipeline_data, error, warning)
        """
        pass

    def initialise(self):
        """
            Called before the transformation method is executed and is primarily to be used to load models
            once.
        """
        pass

    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        """
            Dumps the intermediate data produced by this transformation for debugging purposes into a directory named
            after the class that implements `dump_intermediate_output`. Dumped data includes images that show areas of
            interest and dumps of intermediate processing values.

            NOTE: Outputs from previous runs may be present in the output directory. Stages are responsible for ensuring
            that the outputs they write will handle such a case, either by overwriting files (the default for OpenCV and
            Python's open() file I/O) or deleting them first.

            :param stage_directory: The directory to store the output for the current stage.
            :param pipeline_output: Dictionary object that is the final output for the pipeline
            :param pipeline_data: Object that stores the intermediate data produced by each stage of the pipeline.
            `transform` is called first so pipeline_data contains new values added by that method.
            :return: Returned values are ignored.
        """
        pass


class PhotoExtraction(Stage):
    """
    Creates a hash of the image and extracts the photo timestamp if it is present. Both the hash and the timestamp
    are added to the pipeline_output object.
    """

    def operate(self, pipeline_output, pipeline_data):
        error = None
        warning = None
        try:
            numpy_image = cv2.imdecode(np.fromstring(pipeline_data.image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if numpy_image is None:
                raise RuntimeError("Invalid image")

            # Extract image metadata.
            pipeline_output["imageWidth"] = numpy_image.shape[1]
            pipeline_output["imageHeight"] = numpy_image.shape[0]
            pipeline_output["photoHash"] = file_utils.md5_from_bytes(pipeline_data.image_bytes)
            pipeline_data.image_array = numpy_image
            pipeline_data.image_rgb_array = numpy_image[...,::-1]

            # Extract timestamp data from EXIF data
            if imghdr.what(pipeline_data.image_filename, pipeline_data.image_bytes) == 'jpeg':
                pipeline_data.image_exif = piexif.load(pipeline_data.image_bytes)
                if pipeline_data.image_exif["Exif"] and \
                        piexif.ExifIFD.DateTimeOriginal in pipeline_data.image_exif["Exif"]:
                    timestamp = pipeline_data.image_exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode('UTF-8')
                    pipeline_output["photoTimestamp"] = timestamp
        except struct.error:
            warning = WarningCode.INVALID_EXIF_DATA
            LOGGER.warning("Invalid EXIF Data provided")
        except RuntimeError as e:
            error = ErrorCode.INVALID_FILE

        return pipeline_output, pipeline_data, error, warning


class DownsampleImage(Transformation):
    def transform(self, pipeline_output, pipeline_data):
        config = get_config()
        error = None
        max_size = config["imageSizeMax"]
        image = pipeline_data.image_array

        new_width = 0
        new_height = 0
        width = pipeline_output["imageWidth"]
        height = pipeline_output["imageHeight"]

        if width > max_size:
            r = max_size / float(width)
            new_height = int(r * height)
            new_width = max_size
        elif height > max_size:
            r = max_size / float(height)
            new_height = max_size
            new_width = int(r * width)

        if new_width > 0 and new_height > 0:
            new_image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
            pipeline_data.image_array = new_image
            pipeline_data.image_rgb_array = new_image[...,::-1]
            pipeline_output["imageWidth"] = new_width
            pipeline_output["imageHeight"] = new_height

        return pipeline_output, pipeline_data, error, None

    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        new_image = pipeline_data.image_array.copy()
        file_name = os.path.splitext(basename(pipeline_data.image_filename))[0]
        cv2.imwrite(os.path.join(stage_directory, "downsampled-%s.jpg" % file_name), new_image)


class RotateImage(Transformation):
    """
    Detect the orientation of an image and rotate so that the face is always upright
    """

    rotation_angles = dict(
        upright=0,
        left=270,
        upsidedown=180,
        right=90
    )

    def normalise(self, image):
        config = get_config()
        input_height = config["rotateImageInputHeight"]
        input_width = config["rotateImageInputWidth"]
        with tf.Graph().as_default():
            # Convert OpenCV Numpy array to a Tensorflow tensor
            # Note that there is a slight error between an OpenCV decoded image and a Tensorflow decoded image.
            convert = tf.convert_to_tensor(image, dtype=tf.float32)
            dims_expander = tf.expand_dims(convert, 0)
            resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
            normalized = tf.multiply(resized, 1./255)
            sess = tf.Session()
            result = sess.run(normalized)
        return result

    def initialise(self):
        config = get_config()
        self.labels = self.load_labels(os.path.join(config["pathToModels"],config["rotateImageModelLabels"]))

        model_path = os.path.join(config["pathToModels"], config["rotateImageModelFile"])
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()

        self.has_model = os.path.exists(model_path)
        if not self.has_model:
            LOGGER.error("No model found at %s" % model_path)
        else:

            with open(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
            with self.graph.as_default():
                 tf.import_graph_def(graph_def)
            self.session = tf.Session(graph=self.graph)


    def rotate_and_scale(self, img, scaleFactor = 1, degreesCCW = 30):
        """
        Rotate and scale an image without transposing the matrix. The CNN requires an image to be in a certain
        format so transposed images need to be preprocessed again before being passed to the network.
        """
        (oldY,oldX, _) = img.shape
        M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

        #choose a new image size.
        newX,newY = oldX*scaleFactor,oldY*scaleFactor
        #include this if you want to prevent corners being cut off
        r = np.deg2rad(degreesCCW)
        newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

        #So I will find the translation that moves the result to the center of that region.
        (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
        M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
        M[1,2] += ty

        rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
        return rotatedImg

    def transform(self, pipeline_output, pipeline_data):
        error = None
        warning = None
        config = get_config()

        if config["rotateImageSkip"]:
            return pipeline_output, pipeline_data, error, warning

        if self.has_model:
            input_name = "import/" + config["rotateImageInputLayer"]
            output_name = "import/" + config["rotateImageOutputLayer"]
            normalised_image = self.normalise(pipeline_data.image_rgb_array)

            input_operation = self.graph.get_operation_by_name(input_name)
            output_operation = self.graph.get_operation_by_name(output_name)

            self.results = self.session.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: normalised_image
                })
            self.results = np.squeeze(self.results)
            self.top = self.results.argsort()
            index = self.top[-1]

            if self.results[index] > config["rotateImageThreshold"]:
                label = self.labels[index]
                angle = self.rotation_angles[label]

                if not label == "upright":
                    pipeline_data.image_array = self.rotate_and_scale(pipeline_data.image_array, degreesCCW=angle)
                    pipeline_data.image_rgb_array = pipeline_data.image_array[...,::-1]
                    warning = WarningCode.IMAGE_ROTATED
            else:
                warning = WarningCode.IMAGE_ROTATION_UNDEFINED
        else:
            error = ErrorCode.NO_MODEL_ROTATION
        return pipeline_output, pipeline_data, error, warning


    def load_labels(self, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label


    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        config = get_config()
        if config["rotateImageSkip"]:
            return

        file_name = os.path.splitext(basename(pipeline_data.image_filename))[0]
        cv2.imwrite(os.path.join(stage_directory, "faces-flipped-%s.jpg" % file_name), pipeline_data.image_array)
        if self.has_model:
            with open(os.path.join(stage_directory, "rotation-%s.txt" % file_name), "a") as f:
                for i in self.top:
                    f.write("%s: %f\n" % (self.labels[i], self.results[i]))


class LargestFace(Transformation):
    """
        Find the bounding box for the largest face in the image.
    """

    def transform(self, pipeline_output, pipeline_data):
        # Find faces in the image
        config = get_config()
        error = None
        warning = None
        max_area = 0
        faces = list()
        a_face_too_small = False
        if len(pipeline_data.face_bounding_boxes) == 0:
            error = ErrorCode.FACE_NOT_FOUND
        else:

            # @TODO: Vectorise finding the largest bounding box for performance improvements
            # Find the largest face in the image
            for i, box in enumerate(pipeline_data.face_bounding_boxes):

                # box is a tuple in the format (x1, y1, x2, y2)
                x1 = self.restrict_value_along_axis(box[0], pipeline_data.image_array.shape[1])
                y1 = self.restrict_value_along_axis(box[1], pipeline_data.image_array.shape[0])
                x2 = self.restrict_value_along_axis(box[2], pipeline_data.image_array.shape[1])
                y2 = self.restrict_value_along_axis(box[3], pipeline_data.image_array.shape[0])

                width = x2 - x1
                height = y2 - y1
                area = abs(width * height)
                new_box = (x1, y1, x2, y2)

                if config["useAllFaces"]:
                    faces.append(new_box)
                    if not a_face_too_small:
                        a_face_too_small = width < config["minFaceWidth"] or height < config["minFaceHeight"]
                else:
                    if area > max_area:
                        max_area = area
                        faces = [new_box]
                        a_face_too_small = width < config["minFaceWidth"] or height < config["minFaceHeight"]

            if len(faces) == 0 and a_face_too_small:
                warning = WarningCode.FACE_TOO_SMALL

            else:
                pipeline_data.face_filtered_boxes = faces
                pipeline_output["faceBoundingBoxes"] = [ list(box) for box in faces ]
                pipeline_output["faceBoundingBoxes"] = pipeline_output["faceBoundingBoxes"][0];
        return pipeline_output, pipeline_data, error, warning

    def restrict_value_along_axis(self, value, max):
        if value < 0:
            new_value = 0
        elif value > max:
            new_value = max - 1
        else:
            new_value = value
        return new_value

    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        new_image = pipeline_data.image_array.copy()

        if not pipeline_data.face_filtered_boxes or len(pipeline_data.face_filtered_boxes) == 0:
            logging.error("No largest face available")
            return

        for face in pipeline_data.face_filtered_boxes:
            # Draw bounding box around largest face
            cv2.rectangle(new_image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 3)

        file_name = os.path.splitext(basename(pipeline_data.image_filename))[0]
        cv2.imwrite(os.path.join(stage_directory, "largest-face-%s.jpg" % file_name), new_image)


class ImageTooDark(Transformation):
    """
    Determines if an image is too dark for further processing.
    """
    def transform(self, pipeline_output, pipeline_data):
        warning = None
        config = get_config()
        self.darkness_values = list()
        pipeline_data.image_grayscale = cv2.cvtColor(pipeline_data.image_array, cv2.COLOR_BGR2GRAY)
        avg = pipeline_data.image_grayscale.mean()
        pipeline_output["grayscaleMean"] = avg
        self.darkness_values.append(avg)
        if avg < config["imageTooDark"]:
            warning = WarningCode.IMAGE_TOO_DARK
        return pipeline_output, pipeline_data, None, warning

    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        file_name = os.path.splitext(basename(pipeline_data.image_filename))[0]

        for i, value in enumerate(self.darkness_values):
            with open(os.path.join(stage_directory, "darkness-%s-%d.txt" % (file_name, i)), "w") as f:
                f.write("%d, %s\n" % (value, pipeline_data.image_filename))


class FaceTooBlurry(Transformation):
    """
    Reject images where the face(s) found are too blurry for further processing.
    """
    def transform(self, pipeline_output, pipeline_data):
        config = get_config()
        warning = None
        blurry_face_found = False
        at_least_one_face_in_focus = False
        self.blur_values = list()
        for i, face in enumerate(pipeline_data.face_filtered_boxes):
            face_image = pipeline_data.image_array[face[1]:face[3], face[0]:face[2]]
            gray_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            value = cv2.Laplacian(gray_face_image, cv2.CV_64F).var()
            if value < config["blurryThreshold"]:
                blurry_face_found = True
            else:
                at_least_one_face_in_focus = True
            self.blur_values.append(value)

        if not at_least_one_face_in_focus and blurry_face_found:
            warning = WarningCode.FACE_TOO_BLURRY

        return pipeline_output, pipeline_data, None, warning

    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        file_name = os.path.splitext(basename(pipeline_data.image_filename))[0]

        for i, value in enumerate(self.blur_values):
            with open(os.path.join(stage_directory, "blur-%s-%d.txt" % (file_name, i)), "w") as f:
                f.write("%d, %s\n" % (value, pipeline_data.image_filename))


class ExtractEncodingsResNet1(Transformation):

    @staticmethod
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def initialise(self):
        config = get_config()

        with tf.Graph().as_default():
            # Configure GPU settings.
            # @TODO: Support loading full GPU options dict from config. This would require supporting nested environment overrides (see utils.py).
            if config["gpuDynamicMemoryAllocation"]:
                LOGGER.info("Tensorflow will use dynamic GPU memory allocation.")
                gpu_options = tf.GPUOptions(allow_growth=True)
            else:
                LOGGER.info("Tensorflow will use preconfigured GPU memory fraction of {}".format(config["perProcessGpuMemoryFraction"]))
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config["perProcessGpuMemoryFraction"])

            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                model_path = os.path.join(config["pathToModels"], "20170512-110547", "20170512-110547.pb")
                self.has_model = os.path.exists(model_path)

                if not self.has_model:
                    LOGGER.error("No model found at %s" % model_path)
                else:

                    with gfile.FastGFile(model_path,'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        tf.import_graph_def(graph_def, name='')

                        # Get input and output tensors
                        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def transform(self, pipeline_output, pipeline_data):
        error = None
        self.resized_images = list()

        if self.has_model:
            image_size = 160

            # Create data structure for holding the image that is expected by the Neural Network
            images = np.zeros((len(pipeline_data.face_filtered_boxes), image_size, image_size, 3))

            for i, face in enumerate(pipeline_data.face_filtered_boxes):

                # Crop and resize image
                image = pipeline_data.image_array[face[1]:face[3], face[0]:face[2]]
                resized = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_LINEAR)
                self.resized_images.append(resized)
                images[i,:,:,:] = self.prewhiten(resized)

                # Extract encodings
            feed_dict = {self.images_placeholder:images, self.phase_train_placeholder:False }
            pipeline_data.face_encodings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        else:
            error = ErrorCode.NO_MODEL_ENCODING

        return pipeline_output, pipeline_data, error, None

    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        file_name = os.path.splitext(basename(pipeline_data.image_filename))[0]
        for i, image in enumerate(self.resized_images):
            cv2.imwrite(os.path.join(stage_directory, "resized-cropped-face-%s-%d.jpg" % (file_name, i)), image)

class DetectAndAlignFaces(Transformation):

    pnet = None
    rnet = None
    onet = None

    def initialise(self):
        config = get_config()

        with tf.Graph().as_default():
            # Configure GPU settings.
            # @TODO: Support loading full GPU options dict from config. This would require supporting nested environment overrides (see utils.py).
            if config["gpuDynamicMemoryAllocation"]:
                LOGGER.info("Tensorflow will use dynamic GPU memory allocation.")
                gpu_options = tf.GPUOptions(allow_growth=True)
            else:
                LOGGER.info("Tensorflow will use preconfigured GPU memory fraction of {}".format(config["perProcessGpuMemoryFraction"]))
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config["perProcessGpuMemoryFraction"])

            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                model_path = os.path.join(config["pathToModels"], "Multi-task-CNN")
                self.has_model = os.path.exists(model_path)

                if not self.has_model:
                    LOGGER.error("No model found at %s" % model_path)
                else:
                    self.pnet, self.rnet, self.onet = create_mtcnn(sess, model_path)


    def look_for_faces(self, image):
        """
        Use the Multi-task CNN to find and align faces in `image`
        """
        config = get_config()
        minWidth = config["minFaceWidth"]
        minHeight = config["minFaceHeight"]
        minsize = minWidth if minWidth < minHeight else minHeight

        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor

        # @TODO: Parametise the image size embedded in Facenet
        boxes, points = detect_face(image, minsize, self.pnet, self.rnet, self.onet, threshold, factor)

        bounding_boxes = []
        for i in range(boxes.shape[0]):
            positions = [int(a) for a in boxes[i]]
            bounding_boxes.append((positions[0],positions[1],positions[2], positions[3]))
        return bounding_boxes

    def transform(self, pipeline_output, pipeline_data):

        error = None
        warning = None
        assert not self.pnet is None and not self.rnet is None and not self.onet is None, "NN not setup correctly"
        if self.has_model:
            bounding_boxes = self.look_for_faces(pipeline_data.image_rgb_array)
            pipeline_data.face_bounding_boxes = bounding_boxes
        else:
            error = ErrorCode.NO_MODEL_FACE_DETECTOR
        return pipeline_output, pipeline_data, error, warning

    def dump_intermediate_output(self, stage_directory, pipeline_output, pipeline_data):
        LOGGER.debug("Bounding boxes found by DetectAndAlign")
        LOGGER.debug(pipeline_data.face_bounding_boxes)
        new_image = pipeline_data.image_array.copy()
        file_name = os.path.splitext(basename(pipeline_data.image_filename))[0]

        if len(pipeline_data.face_bounding_boxes) == 0:
            LOGGER.info("No bounding boxes available")
            return

        for i, box in enumerate(pipeline_data.face_bounding_boxes):
            cv2.rectangle(new_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            cv2.putText(new_image,str(i),(box[0]+5, box[3]-5), cv2.FONT_HERSHEY_SIMPLEX, 4,(0, 255, 0),2,cv2.LINE_AA)

        cv2.imwrite(os.path.join(stage_directory, "faces-%s.jpg" % file_name), new_image)



class BuildOutput(Transformation):
    def transform(self, pipeline_output, pipeline_data):
        image_name = os.path.splitext(basename(pipeline_data.image_filename))[0]

        if matches_expected_naming_format(image_name):
            associated_id = image_name.split('_')[0]
        else:
            associated_id = "UNKNOWN"

        pipeline_output["associatedId"] = associated_id
        pipeline_output["faceEncodings"] = pipeline_data.face_encodings.tolist()
        pipeline_output["faceEncodings"] = pipeline_output["faceEncodings"][0]
        return pipeline_output, pipeline_data, None, None


class Pipeline(object):

    def __init__(self):
        self.build_output = BuildOutput()
        self.pipeline_stages = [
            PhotoExtraction(),
            DownsampleImage(),
            RotateImage(),
            ImageTooDark(),
            DetectAndAlignFaces(),
            LargestFace(),
            FaceTooBlurry(),
            ExtractEncodingsResNet1(),
            self.build_output
        ]

    def initialise_pipeline(self):
        for i in self.pipeline_stages:
            i.initialise()

    def process_photo(self, image_file, input_directory, output_directory):
        LOGGER.info("test");
        """
        Runs the image processing pipeline on the provided image file to extract a face encoding.
        """
        return self.process_photo_bytes(image_file, open(image_file, "rb").read(), input_directory, output_directory)


    def process_photo_bytes(self, image_filename, image_bytes, input_directory, output_directory):
        """
        Runs the image processing pipeline on the provided image data to extract a face encoding.

        Returns a tuple of (output, error) where the output is a dictionary of attributes of a photo and error is an
        error code specified in utils.ErrorCode.
        """
        error = None
        warning = None
        

        config = get_config()



        # Output from processing a photo
        # @TODO: Convert this to a PipelineOutput class
        pipeline_output = dict(photoFilename=image_filename)
        

        stage_input = PipelineData()


        stage_input.image_filename = image_filename
        stage_input.image_bytes = image_bytes


        warnings = list()


        for stage in self.pipeline_stages:
            try:
                stage_start = None
                if config["printStageProcessingTime"]:
                    stage_start = datetime.now()

                # The output of a stage becomes the input of the next stage
                pipeline_output, stage_output, error, warning = stage.transform(pipeline_output, stage_input)
                assert isinstance(stage_output, PipelineData), "Output must be a PipelineData object"

                if config["enableStageOutputDump"]:
                    # Create directory for intermediate stage output
                    stage_directory = os.path.join(output_directory, type(stage).__name__)
                    if not os.path.exists(stage_directory):
                        os.makedirs(stage_directory)

                    stage.dump_intermediate_output(stage_directory, pipeline_output, stage_output)

                stage_input = stage_output
                assert not error or error in error_values(), \
                    "Invalid error, error value must map to an error variable in the ErrorCode class"

                # Calculate and log stage duration (if enabled).
                if config["printStageProcessingTime"]:
                    stage_end = datetime.now()
                    stage_duration = stage_end - stage_start
                    LOGGER.info("Stage {} took {} secs".format(type(stage).__name__, stage_duration))
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                filename, line, function, text = traceback.extract_tb(exc_traceback)[-1]

                error = ErrorCode.UNKNOWN_ERROR(
                    error_name=exc_type.__name__,
                    error_message=str(exc_value),
                    file=filename,
                    line=line
                )

                LOGGER.exception("Error processing %s", image_filename)
            if error:

                # Create a random encoding when an error occurs and config parameter set
                if config["alwaysCreateEncoding"] and not error == ErrorCode.INVALID_FILE:
                    warnings.append(WarningCode.CREATED_ENCODING)
                    stage_input.face_encodings = np.array([np.random.rand(128)])
                    pipeline_output, _ , _, _ = self.build_output.transform(pipeline_output, stage_input)
                else:
                    break
            if warning:
                warnings.append(warning)

        return pipeline_output, error, warnings