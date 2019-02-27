from imutils.video import WebcamVideoStream
import cv2
import numpy as np
import os
import logging
import base64
from surround import Config

LOGGER = logging.getLogger(__name__)

# Load config file.
CONFIG = Config()
CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
CONFIG.read_config_files([CONFIG_PATH])


class FaceDetectionWebcamStream(WebcamVideoStream):
	"""
	Starts a separate thread to capture frames from the webcam,
	performs per-frame face detection, and stores latest frame along with
	a bounding box.
	"""
	def __init__(self, src=0, name="FaceDetectionWebcamStream"):
		super().__init__(src, name)
		self.src = src
		self.grabbed = False
		self.boxes = []
		self.net = cv2.dnn.readNetFromCaffe(
			os.path.dirname(os.path.realpath(__file__)) + "/../models/deploy.prototxt.txt",
			os.path.dirname(os.path.realpath(__file__)) + "/../models/res10_300x300_ssd_iter_140000.caffemodel")

	def update(self):
		"""
		Update loop that runs on its own thread to grab frames from the webcam
		and perform face detection.
		"""
		# Keep looping infinitely until the thread is stopped.
		while True:
			if self.stopped:
				return

			if self.stream is None or not self.stream.isOpened():
				LOGGER.error("Camera device {} not found. Stopping stream.".format(self.src))
				self.stop()
				return

			# Read the next frame from the stream.
			(grabbed, frame) = self.stream.read()

			# Grab the frame dimensions and convert it to a blob.
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
				(300, 300), (104.0, 177.0, 123.0))

			# Pass the blob through the network and obtain the detections and predictions.
			self.net.setInput(blob)
			detections = self.net.forward()

			boxes = []
			for i in range(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]

				# Filter out weak detections.
				if confidence < CONFIG["webcamStream"]["minConfidence"]:
					continue

				# Compute the (x, y)-coordinates of the bounding box.
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				boxes.append((startX, startY, endX, endY))

				if CONFIG["webcamStream"]["drawBox"]:
					# Draw the bounding box of the face along with the associated probability.
					text = "{:.2f}%".format(confidence * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10

					if confidence >= CONFIG["webcamStream"]["highConfidence"]:
						colour = (0, 255, 0)
					else:
						colour = (0, 0, 255)

					cv2.rectangle(frame, (startX, startY), (endX, endY), colour)
					cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

			# Update data for clients to consume.
			# @TODO: Thread safety
			self.frame = frame
			self.boxes = boxes
			result, bmp = cv2.imencode(".bmp", frame)
			b64 = base64.b64encode(bmp)
			self.frame_base64 = b64
			self.grabbed = grabbed

	def read_base64(self):
		return self.frame_base64
