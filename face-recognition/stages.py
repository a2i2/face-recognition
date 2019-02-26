import os
import cv2
import logging
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from .align import create_mtcnn, detect_face
from surround import Stage, SurroundData, Surround, Config


LOGGER = logging.getLogger(__name__)


def face_recognition_pipeline():
    """
    Returns the stages that form part of the face recognition pipeline.
    """
    return Surround([
        DecodeImage(),
        DownsampleImage(),
        RotateImage(),
        ImageTooDark(),
        DetectAndAlignFaces(),
        LargestFace(),
        FaceTooBlurry(),
        ExtractFaceEncodings()
    ])


class FaceRecognitionPipelineData(SurroundData):
    """
    Stores the data to be passed between each stage of a pipeline. Note that different stages of the pipeline are
    responsible for setting the attributes.
    """
    image_array = None            # Numpy array of the image stored in BGR format
    image_rgb_array = None        # Image in RGB format
    image_filename = None         # File name of the image
    image_bytes = None            # Image bytes. If this is populated, the DecodeImage stage will load from here rather than image_filename.
    image_exif = None             # Exif data for the image
    image_grayscale = None        # Grayscale version of the image
    face_bounding_boxes = list()  # List of tuples, (x1, y1, x2, y2) bounding boxes for faces found in the image
                                  # where (x1,y1) is the top left point at (x2, y2) is the bottom right point.
    face_filtered_boxes = list()  # Filtered list of bounding boxes of faces in the image in the form (x1, y1, x2, y2)
    face_encodings = list()       # List of 128 floats representing a face encoding
    output_data = dict(photoFilename=image_filename)

    def __init__(self, image_filename, image_bytes=None):
        self.image_filename = image_filename
        self.image_bytes = image_bytes
        self.output_data["photoFilename"] = image_filename


class DecodeImage(Stage):
    """
    Loads image data from the provided bytes or file path
    and decodes it into into a numpy array for input to the rest of the pipeline.
    """
    def operate(self, surround_data, config):
        if surround_data.image_bytes is None:
            try:
                surround_data.image_bytes = open(surround_data.image_filename, "rb").read()
            except:
                surround_data.error = dict(error="No such image file")
                return

        numpy_image = cv2.imdecode(np.fromstring(surround_data.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if numpy_image is None:
            surround_data.error = dict(error="Error decoding image")
            return

        surround_data.image_array = numpy_image
        surround_data.image_rgb_array = numpy_image[...,::-1]


class DownsampleImage(Stage):
    """
    Downsamples the image if it is larger than the configured width/height limit.
    """
    def operate(self, surround_data, config):
        max_size = config["imageSizeMax"]
        image = surround_data.image_array

        new_width = 0
        new_height = 0
        width = image.shape[1]
        height = image.shape[0]

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
            surround_data.image_array = new_image
            surround_data.image_rgb_array = new_image[...,::-1]


class RotateImage(Stage):
    """
    Detects the orientation of the image and rotates it if necessary. Combats issues
    that arise when EXIF orientation data is missing.
    """
    rotation_angles = dict(
        upright=0,
        left=270,
        upsidedown=180,
        right=90)

    def normalise(self, image, config):
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

    def load_labels(self, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for line in proto_as_ascii_lines:
            label.append(line.rstrip())
        return label

    def init_stage(self, config):
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), config["pathToModels"])
        self.labels = self.load_labels(os.path.join(models_dir, config["rotateImageModelLabels"]))
        model_path = os.path.join(os.path.join(models_dir, config["rotateImageModelFile"]))
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()

        self.has_model = os.path.exists(model_path)
        if not os.path.exists(model_path):
            LOGGER.error("No model found at %s" % model_path)
        else:
            with open(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
            with self.graph.as_default():
                 tf.import_graph_def(graph_def)
            self.session = tf.Session(graph=self.graph)

    def rotate_and_scale(self, image, scale_factor=1, degrees_ccw=30):
        """
        Rotate and scale an image without transposing the matrix. The CNN requires an image to be in a certain
        format so transposed images need to be preprocessed again before being passed to the network.
        """
        (old_y, old_x, _) = image.shape

        # Rotate about center of image.
        rotation_matrix = cv2.getRotationMatrix2D(center=(old_x/2, old_y/2), angle=degrees_ccw, scale=scale_factor)

        # Choose a new image size.
        new_x, new_y = old_x * scale_factor, old_y * scale_factor

        # Prevent corners being cut off.
        r = np.deg2rad(degrees_ccw)
        new_x,new_y = (abs(np.sin(r)*new_y) + abs(np.cos(r)*new_x),abs(np.sin(r)*new_x) + abs(np.cos(r)*new_y))

        # Find the translation that moves the result to the center of that region.
        (tx, ty) = ((new_x - old_x) / 2, (new_y - old_y) / 2)

        rotation_matrix[0,2] += tx  # Third column of matrix holds translation, which takes effect after rotation.
        rotation_matrix[1,2] += ty

        rotatedImg = cv2.warpAffine(image, rotation_matrix, dsize=(int(new_x),int(new_y)))
        return rotatedImg

    def operate(self, surround_data, config):
        if config["rotateImageSkip"]:
            return

        if self.has_model:
            input_name = "import/" + config["rotateImageInputLayer"]
            output_name = "import/" + config["rotateImageOutputLayer"]
            normalised_image = self.normalise(surround_data.image_rgb_array, config)

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
                    surround_data.image_array = self.rotate_and_scale(surround_data.image_array, degrees_ccw=angle)
                    surround_data.image_rgb_array = surround_data.image_array[...,::-1]
                    surround_data.warnings.append("IMAGE_ROTATED")
            else:
                surround_data.warnings.append("IMAGE_ROTATION_UNDEFINED")
        else:
            surround_data.error = dict(error="NO_MODEL_FOR_FACE_ROTATION")
            return


class ImageTooDark(Stage):
    """
    Analyses the image and produces a warning if it is deemed
    to be too dark to produce an accurate face encoding.
    """
    def operate(self, surround_data, config):
        self.darkness_values = list()
        surround_data.image_grayscale = cv2.cvtColor(surround_data.image_array, cv2.COLOR_BGR2GRAY)
        avg = surround_data.image_grayscale.mean()
        self.darkness_values.append(avg)
        if avg < config["imageTooDark"]:
            surround_data.warnings.append("IMAGE_TOO_DARK")


class DetectAndAlignFaces(Stage):
    """
    Detects faces in the image and produces a bounding box for each.
    """
    def init_stage(self, config):
        with tf.Graph().as_default():
            # Configure GPU settings.
            if config["gpuDynamicMemoryAllocation"]:
                gpu_options = tf.GPUOptions(allow_growth=True)
            else:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config["perProcessGpuMemoryFraction"])

            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), config["pathToModels"])
                model_path = os.path.join(models_dir, "Multi-task-CNN")
                self.has_model = os.path.exists(model_path)

                if not self.has_model:
                    LOGGER.error("No model found at %s" % model_path)
                else:
                    self.pnet, self.rnet, self.onet = create_mtcnn(sess, model_path)

    def look_for_faces(self, image, config):
        """
        Use the Multi-task CNN to find and align faces in the given image.
        """
        minWidth = config["minFaceWidth"]
        minHeight = config["minFaceHeight"]
        minsize = minWidth if minWidth < minHeight else minHeight

        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor

        # @TODO: Parameterise the image size embedded in Facenet
        boxes, points = detect_face(image, minsize, self.pnet, self.rnet, self.onet, threshold, factor)

        bounding_boxes = []
        for i in range(boxes.shape[0]):
            positions = [int(a) for a in boxes[i]]
            bounding_boxes.append((positions[0],positions[1],positions[2], positions[3]))
        return bounding_boxes

    def operate(self, surround_data, config):
        assert not self.pnet is None and not self.rnet is None and not self.onet is None, "NN not set up correctly"
        if self.has_model:
            bounding_boxes = self.look_for_faces(surround_data.image_rgb_array, config)
            surround_data.face_bounding_boxes = bounding_boxes
        else:
            surround_data.error = dict(error="NO_MODEL_FOR_FACE_DETECTION")
            return


class LargestFace(Stage):
    """
    If configured to do so, filter the output data to only consider
    the largest face.
    """
    def operate(self, surround_data, config):
        max_area = 0
        faces = list()
        a_face_too_small = False
        if len(surround_data.face_bounding_boxes) == 0:
            surround_data.error = dict(error="FACE_NOT_FOUND")
            return
        else:
            for i, box in enumerate(surround_data.face_bounding_boxes):

                # box is a tuple in the format (x1, y1, x2, y2)
                x1 = self.restrict_value_along_axis(box[0], surround_data.image_array.shape[1])
                y1 = self.restrict_value_along_axis(box[1], surround_data.image_array.shape[0])
                x2 = self.restrict_value_along_axis(box[2], surround_data.image_array.shape[1])
                y2 = self.restrict_value_along_axis(box[3], surround_data.image_array.shape[0])

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
                surround_data.warnings.append("FACE_TOO_SMALL")
            else:
                surround_data.face_filtered_boxes = faces

    def restrict_value_along_axis(self, value, max):
        if value < 0:
            new_value = 0
        elif value > max:
            new_value = max - 1
        else:
            new_value = value
        return new_value


class FaceTooBlurry(Stage):
    """
    Raise a warning if the face is deemed to be too blurry.
    """
    def operate(self, surround_data, config):
        blurry_face_found = False
        at_least_one_face_in_focus = False
        self.blur_values = list()
        for i, face in enumerate(surround_data.face_filtered_boxes):
            face_image = surround_data.image_array[face[1]:face[3], face[0]:face[2]]
            gray_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            value = cv2.Laplacian(gray_face_image, cv2.CV_64F).var()
            if value < config["blurryThreshold"]:
                blurry_face_found = True
            else:
                at_least_one_face_in_focus = True
            self.blur_values.append(value)

        if not at_least_one_face_in_focus and blurry_face_found:
            surround_data.warnings.append("FACE_TOO_BLURRY")


class ExtractFaceEncodings(Stage):
    """
    Extracts a face encoding for each detected face in the image.
    This encoding can be compared with other encodings using a euclidean distance measure
    to determine how similar the faces are.
    """
    @staticmethod
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def init_stage(self, config):
        with tf.Graph().as_default():
            # Configure GPU settings.
            if config["gpuDynamicMemoryAllocation"]:
                gpu_options = tf.GPUOptions(allow_growth=True)
            else:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config["perProcessGpuMemoryFraction"])

            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), config["pathToModels"])
                model_path = os.path.join(models_dir, "20170512-110547", "20170512-110547.pb")
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

    def operate(self, surround_data, config):
        self.resized_images = list()

        if self.has_model:
            image_size = 160

            # Create data structure for holding the image that is expected by the neural network.
            images = np.zeros((len(surround_data.face_filtered_boxes), image_size, image_size, 3))

            for i, face in enumerate(surround_data.face_filtered_boxes):
                # Crop and resize image.
                image = surround_data.image_array[face[1]:face[3], face[0]:face[2]]
                resized = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_LINEAR)
                self.resized_images.append(resized)
                images[i,:,:,:] = self.prewhiten(resized)

            # Extract encodings.
            feed_dict = {self.images_placeholder:images, self.phase_train_placeholder:False}
            face_encodings = self.sess.run(self.embeddings, feed_dict=feed_dict).tolist()
            surround_data.face_encodings = face_encodings
            surround_data.output_data["face_encodings"] = face_encodings
        else:
            surround_data.error = dict(error="NO_MODEL_FOR_FACE_ENCODING")
            return
