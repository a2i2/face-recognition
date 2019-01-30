import logging

from abc import ABC, abstractmethod
from dstil.queue.rabbit_binder import RabbitBinder


class Processor(ABC):
    """
    Analogous to a Spring Stream 'Processor' that listens on one queue, does some processing,
    and sends the output to another queue.

    This class is abstract. Users must implement the 'handle_message' method and return a value
    that can be serialised to JSON.

    @TODO:
    * Define input/output contracts for messages and ensure return types are not None.
    * Implement proper error handling.
    """

    def __init__(self, host, port, username, password, input_binding, output_binding):
        self.logger = logging.getLogger(__name__)
        self.binder = RabbitBinder(
            host,
            port,
            username,
            password,
            input_binding,
            output_binding,
            self.__handle_message_internal
        )

    def __handle_message_internal(self, channel, method, properties, body):
        """
        Handles incoming messages.
        :param channel: The message channel on which the message was received.
        :param method: Delivery details of the message.
        :param properties: Message metadata (headers, etc.)
        :param body: The message itself.
        """
        self.logger.debug("Received message with properties: %r" % properties)
        self.logger.debug(body)

        try:
            result = self.handle_message(properties, body)

            # Copy request headers in response.
            response_headers = properties.headers
            response_headers["contentType"] = "application/json"

            # Publish to output queue.
            if result is not None:
                self.binder.publish_message(response_headers, result)
            else:
                self.logger.warning("Processor returned None; not sending anything to the output queue")
        except Exception as e:
            # Don't crash if an error occurs.
            # @TODO: Implement sensible error handling.
            self.logger.exception(e)

    @abstractmethod
    def handle_message(self, properties, body):
        """
        User-defined message handler. Must be implemented by subclasses.
        :return:
        """
        pass

    def listen(self):
        """
        Starts listening on the input queue for messages.
        This is a blocking call.
        :return:
        """
        try:
            self.binder.run()
        except KeyboardInterrupt:   # @TODO: Handle SIGINT/SIGTERM instead
            self.clean_up()

    def clean_up(self):
        """
        Cleans up the RabbitMQ binder and closes the connection.
        :return:
        """
        self.binder.stop()
        self.binder.close_connection()
