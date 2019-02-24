from tornado import gen
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.tcpserver import TCPServer
from .face_detection_webcam_stream import FaceDetectionWebcamStream
import imutils
import pickle
import cv2
import time
import json
import numpy as np
import tornado.ioloop


class WebcamServer(TCPServer):
    """
    This is a simple echo TCP Server
    """
    message_separator = b'\r\n'

    def __init__(self, *args, **kwargs):
        self._connections = []
        super().__init__(*args, **kwargs)

        # Start webcam capture on separate thread
        self.vs = FaceDetectionWebcamStream(src=0)
        self.vs.start()

    def __del__(self):
        if self.vs:
            self.vs.stop()

    @gen.coroutine
    def handle_stream(self, stream, address):
        """
        Main connection loop. Launches listen on given channel and keeps
        reading data from socket until it is closed.
        """
        try:
            print("New connection: {}...".format(address))
            num = 0
            while True:
                num += 1
                try:
                    pass
                except StreamClosedError:
                    stream.close(exc_info=True)
                    return
                else:
                    try:
                        frame = self.vs.read()
                        data = pickle.dumps(frame, 0)
                        yield stream.write(data + self.message_separator)
                    except StreamClosedError:
                        print("Stream closed: {}".format(address))
                        stream.close(exc_info=True)
                        return
        except Exception as e:
            if not isinstance(e, gen.Return):
                print(e)
                print("Connection loop has experienced an error.")
