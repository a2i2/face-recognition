from imutils.video import WebcamVideoStream
import cv2
import numpy as np
import os

class FaceDetectionWebcamStream(WebcamVideoStream):

	def __init__(self, src=0, name="FaceDetectionWebcamStream"):
		super().__init__(src, name)
		self.net = cv2.dnn.readNetFromCaffe(
			os.path.dirname(os.path.realpath(__file__)) + "/../models/deploy.prototxt.txt",
			os.path.dirname(os.path.realpath(__file__)) + "/../models/res10_300x300_ssd_iter_140000.caffemodel")

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(grabbed, frame) = self.stream.read()

			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
				(300, 300), (104.0, 177.0, 123.0))

			# pass the blob through the network and obtain the detections and
			# predictions
			self.net.setInput(blob)
			detections = self.net.forward()

			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the
				# prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence < 0.5:
					continue

				# compute the (x, y)-coordinates of the bounding box for the
				# object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the bounding box of the face along with the associated
				# probability
				text = "{:.2f}%".format(confidence * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			self.grabbed = grabbed
			self.frame = frame
