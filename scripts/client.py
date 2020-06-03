import cv2
import sys
import io
import socket
import struct
import time
import pickle
from pickle import UnpicklingError
import zlib

from tornado import gen
from tornado.iostream import StreamClosedError
from tornado.ioloop import IOLoop
from tornado.tcpclient import TCPClient

class Client(TCPClient):
    """
    This is a simple echo TCP Client
    """
    msg_separator = b'\r\n'

    def __init__(self):
        super().__init__()
        self.retry_timeout = 1

    @gen.coroutine
    def run(self, host, port):
        while True:
            try:
                stream = yield self.connect(host, port)
                while True:
                    try:
                        data = yield stream.read_until(self.msg_separator)
                        body = data.rstrip(self.msg_separator)
                        frame = pickle.loads(body, fix_imports=True, encoding="bytes")
                        cv2.imshow("ImageWindow", frame)
                        cv2.waitKey(1)
                    except EOFError:
                        pass
                    except UnpicklingError:
                        pass
                    except StreamClosedError:
                        raise
                    except Exception as e:
                        print("{}: {}".format(type(e).__name__, e))
            except StreamClosedError:
                cv2.destroyAllWindows()
                print("Stream closed. Retrying connection in {} seconds...".format(self.retry_timeout))
                time.sleep(self.retry_timeout)

if __name__ == '__main__':
    Client().run('localhost', 8889)
    print('Connecting to server socket...')
    IOLoop.instance().start()
    print('Socket has been closed.')
