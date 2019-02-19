import sys
import logging
import pika
import uuid

LOGGER = logging.getLogger(__name__)


class RabbitBinder:
    """
    A RabbitMQ connection adapter and queue binder that enables Python applications
    to communicate with Spring Stream apps.

    @TODO: Turn this into a generic AMQP adapter and build stream-specific functionality on top.
    Current things that are 'fixed' and should be parameterised or overridable:
        * Exchange type is always "topic"
        * Routing key is always "#"
        * A single input/output binding is assumed
        * Queues are always exclusive and auto-delete (to match how Spring configures it)

    The connection adapter will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.
    """
    def __init__(self, host, port, username, password, input_binding=None, output_binding=None, input_callback=None):
        """
        Creates a new AMQP connection, passing in the AMQP URL used to
        connect to RabbitMQ.

        :param str amqp_url: The AMQP url to connect with
        """
        self._id = str(uuid.uuid4())
        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._host = host
        self._port = port
        self._credentials = pika.PlainCredentials(username, password)
        self._input_binding = input_binding
        self._output_binding = output_binding
        self._exchange_type = "topic"
        self._input_queue = input_binding + ".python.input." + self._id
        self._output_queue = input_binding + ".python.output." + self._id
        self._routing_key = "#"
        self._input_callback = input_callback

    def connect(self):
        """
        Connects to RabbitMQ, returning the connection handle.
        When the connection is established, the on_connection_open method
        will be invoked by pika.

        :rtype: pika.SelectConnection
        """
        LOGGER.info("Connecting to %s:%d", self._host, self._port)
        return pika.SelectConnection(
            pika.ConnectionParameters(
                host=self._host,
                port=self._port,
                credentials=self._credentials,
                connection_attempts=sys.maxsize,
                heartbeat=20
            ),
            self.on_connection_open,
            self.on_connection_error,
            self.on_connection_closed,
            stop_ioloop_on_close=False
        )

    def on_connection_error(self, unused_connection, error_message):
        """
        Called by pika once the connection to RabbitMQ has
        been established. It passes the handle to the connection object in
        case we need it, but in this case, we'll just mark it unused.

        :type unused_connection: pika.SelectConnection
        """
        LOGGER.warning(error_message)

    def on_connection_open(self, unused_connection):
        """
        Called by pika once the connection to RabbitMQ has
        been established. It passes the handle to the connection object in
        case we need it, but in this case, we'll just mark it unused.

        :type unused_connection: pika.SelectConnection
        """
        LOGGER.info("Connection opened")
        self.add_on_connection_close_callback()
        self.open_channel()

    def add_on_connection_close_callback(self):
        """
        Adds an on close callback that will be invoked by pika
        when RabbitMQ closes the connection to the publisher unexpectedly.
        """
        LOGGER.info("Adding connection close callback")
        self._connection.add_on_close_callback(self.on_connection_closed)

    def on_connection_closed(self, connection, reply_code, reply_text):
        """
        Invoked by pika when the connection to RabbitMQ is closed unexpectedly.
        Since it is unexpected, we will reconnect to RabbitMQ if it disconnects.

        :param pika.connection.Connection connection: The closed connection obj
        :param int reply_code: The server provided reply_code if given
        :param str reply_text: The server provided reply_text if given
        """
        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            LOGGER.warning(
                "Connection closed, reopening in 5 seconds: (%s) %s",
                reply_code,
                reply_text
            )
            self._connection.add_timeout(5, self.reconnect)

    def reconnect(self):
        """
        Will be invoked by the IOLoop timer if the connection is
        closed. See the on_connection_closed method.
        """
        # This is the old connection IOLoop instance, stop its ioloop
        self._connection.ioloop.stop()

        if not self._closing:
            # Create a new connection
            self._connection = self.connect()

            # There is now a new connection, needs a new ioloop to run
            self._connection.ioloop.start()

    def open_channel(self):
        """
        Opens a new channel with RabbitMQ by issuing the Channel.Open RPC
        command. When RabbitMQ responds that the channel is open, the
        on_channel_open callback will be invoked by pika.
        """
        LOGGER.info("Creating a new channel")
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        """
        Invoked by pika when the channel has been opened.
        The channel object is passed in so we can make use of it.

        Since the channel is now open, we'll declare the exchange to use.

        :param pika.channel.Channel channel: The channel object
        """
        LOGGER.info("Channel opened")
        self._channel = channel
        self.add_on_channel_close_callback()

        if self._input_binding is not None:
            LOGGER.info("Declaring exchange %s", self._input_binding)
            self._channel.exchange_declare(
                self.on_input_exchange_declareok,
                self._input_binding,
                self._exchange_type,
                durable=True
            )

        if self._output_binding is not None:
            LOGGER.info("Declaring exchange %s", self._output_binding)
            self._channel.exchange_declare(
                self.on_output_exchange_declareok,
                self._output_binding,
                self._exchange_type,
                durable=True
            )

    def add_on_channel_close_callback(self):
        """
        Tells pika to call the on_channel_closed method if
        RabbitMQ unexpectedly closes the channel.
        """
        LOGGER.info("Adding channel close callback")
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reply_code, reply_text):
        """
        Invoked by pika when RabbitMQ unexpectedly closes the channel.
        Channels are usually closed if you attempt to do something that
        violates the protocol, such as re-declare an exchange or queue with
        different parameters. In this case, we'll close the connection
        to shut down the object.

        :param pika.channel.Channel: The closed channel
        :param int reply_code: The numeric reason the channel was closed
        :param str reply_text: The text reason the channel was closed
        """
        LOGGER.warning("Channel %i was closed: (%s) %s", channel, reply_code, reply_text)
        self._connection.close()

    def on_input_exchange_declareok(self, unused_frame):
        """
        Invoked by pika when RabbitMQ has finished the Exchange.Declare RPC
        command.

        :param pika.Frame.Method unused_frame: Exchange.DeclareOk response frame
        """
        LOGGER.info("Exchange declared")
        LOGGER.info("Declaring queue %s", self._input_queue)
        self._channel.queue_declare(
            self.on_input_queue_declareok,
            self._input_queue,
            exclusive=True,
            auto_delete=True
        )

    def on_output_exchange_declareok(self, unused_frame):
        """
        Invoked by pika when RabbitMQ has finished the Exchange.Declare RPC
        command.

        :param pika.Frame.Method unused_frame: Exchange.DeclareOk response frame
        """
        LOGGER.info("Exchange declared")
        LOGGER.info("Declaring queue %s", self._output_queue)
        self._channel.queue_declare(
            self.on_output_queue_declareok,
            self._output_queue,
            exclusive=True,
            auto_delete=True
        )

    def on_input_queue_declareok(self, unused_method_frame):
        """
        Invoked by pika when the Queue.Declare RPC call made in
        set_up_queue has completed. In this method we will bind the queue
        and exchange together with the routing key by issuing the Queue.Bind
        RPC command. When this command is complete, the on_bindok method will
        be invoked by pika.

        :param pika.frame.Method method_frame: The Queue.DeclareOk frame
        """
        LOGGER.info(
            "Binding %s to %s with %s",
            self._input_binding,
            self._input_queue,
            self._routing_key
        )

        self._channel.queue_bind(
            self.on_input_bindok,
            self._input_queue,
            self._input_binding,
            self._routing_key
        )

    def on_output_queue_declareok(self, method_frame):
        """
        Invoked by pika when the Queue.Declare RPC call made in
        set_up_queue has completed. In this method we will bind the queue
        and exchange together with the routing key by issuing the Queue.Bind
        RPC command. When this command is complete, the on_bindok method will
        be invoked by pika.

        :param pika.frame.Method method_frame: The Queue.DeclareOk frame
        """
        LOGGER.info(
            "Binding %s to %s with %s",
            self._output_binding,
            self._output_queue,
            self._routing_key
        )

        self._channel.queue_bind(
            self.on_output_bindok,
            self._output_queue,
            self._output_binding,
            self._routing_key
        )

    def on_input_bindok(self, unused_frame):
        """
        Invoked by pika when the Queue.Bind method has completed. At this
        point we will start consuming messages by calling start_consuming
        which will invoke the needed RPC commands to start the process.

        :param pika.frame.Method unused_frame: The Queue.BindOk response frame
        """
        LOGGER.info("Queue bound")
        self.start_consuming()

    def on_output_bindok(self, unused_frame):
        """
        Invoked by pika when the Queue.Bind method has completed. At this
        point we will start consuming messages by calling start_consuming
        which will invoke the needed RPC commands to start the process.

        :param pika.frame.Method unused_frame: The Queue.BindOk response frame
        """
        LOGGER.info("Queue bound")
        self.enable_delivery_confirmations()

    def start_consuming(self):
        """
        Sets up the consumer by first calling add_on_cancel_callback
        so that the object is notified if RabbitMQ cancels the consumer.
        It then issues the Basic.Consume RPC command which returns the consumer
        tag that is used to uniquely identify the consumer with RabbitMQ.
        We keep the value to use it when we want to cancel consuming.
        The on_message method is passed in as a callback pika will invoke
        when a message is fully received.
        """
        LOGGER.info("Issuing consumer related RPC commands")
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(
            self.on_message,
            self._input_queue
        )

    def add_on_cancel_callback(self):
        """
        Adds a callback that will be invoked if RabbitMQ cancels the consumer
        for some reason. If RabbitMQ does cancel the consumer,
        on_consumer_cancelled will be invoked by pika.
        """
        LOGGER.info("Adding consumer cancellation callback")
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        """
        Invoked by pika when RabbitMQ sends a Basic.Cancel for a consumer
        receiving messages.

        :param pika.frame.Method method_frame: The Basic.Cancel frame
        """
        LOGGER.info("Consumer was cancelled remotely, shutting down: %r", method_frame)
        if self._channel:
            self._channel.close()

    def on_message(self, unused_channel, basic_deliver, properties, body):
        """
        Invoked by pika when a message is delivered from RabbitMQ. The
        channel is passed for your convenience. The basic_deliver object that
        is passed in carries the exchange, routing key, delivery tag and
        a redelivered flag for the message. The properties passed in is an
        instance of BasicProperties with the message properties and the body
        is the message that was sent.

        :param pika.channel.Channel unused_channel: The channel object
        :param pika.Spec.Basic.Deliver: basic_deliver method
        :param pika.Spec.BasicProperties: properties
        :param str|unicode body: The message body
        """
        LOGGER.info(
            "Received message # %s from %s",
            basic_deliver.delivery_tag,
            properties.app_id
        )

        self.acknowledge_message(basic_deliver.delivery_tag)
        self._input_callback(unused_channel, basic_deliver, properties, body)

    def publish_message(self, headers, message):
        """If the class is not stopping, publish a message to RabbitMQ,
        appending a list of deliveries with the message number that was sent.
        This list will be used to check for delivery confirmations in the
        on_delivery_confirmations method.

        Once the message has been sent, schedule another message to be sent.
        The main reason I put scheduling in was just so you can get a good idea
        of how the process is flowing by slowing down and speeding up the
        delivery intervals by changing the PUBLISH_INTERVAL constant in the
        class.

        """
        if self._channel is None or not self._channel.is_open:
            return

        properties = pika.BasicProperties(
            content_type="application/json",
            headers=headers
        )

        self._channel.basic_publish(
            self._output_binding,
            self._routing_key,
            message,
            properties
        )

        LOGGER.info("Published message")

    def enable_delivery_confirmations(self):
        """Send the Confirm.Select RPC method to RabbitMQ to enable delivery
        confirmations on the channel. The only way to turn this off is to close
        the channel and create a new one.

        When the message is confirmed from RabbitMQ, the
        on_delivery_confirmation method will be invoked passing in a Basic.Ack
        or Basic.Nack method from RabbitMQ that will indicate which messages it
        is confirming or rejecting.
        """
        LOGGER.info("Issuing Confirm.Select RPC command")
        self._channel.confirm_delivery(self.on_delivery_confirmation)

    def on_delivery_confirmation(self, method_frame):
        """Invoked by pika when RabbitMQ responds to a Basic.Publish RPC
        command, passing in either a Basic.Ack or Basic.Nack frame with
        the delivery tag of the message that was published. The delivery tag
        is an integer counter indicating the message number that was sent
        on the channel via Basic.Publish. Here we're just doing house keeping
        to keep track of stats and remove message numbers that we expect
        a delivery confirmation of from the list used to keep track of messages
        that are pending confirmation.

        :param pika.frame.Method method_frame: Basic.Ack or Basic.Nack frame
        """
        confirmation_type = method_frame.method.NAME.split('.')[1].lower()
        LOGGER.info(
            "Received %s for delivery tag: %i",
            confirmation_type,
            method_frame.method.delivery_tag
        )

    def acknowledge_message(self, delivery_tag):
        """
        Acknowledges the message delivery from RabbitMQ by sending a
        Basic.Ack RPC method for the delivery tag.

        :param int delivery_tag: The delivery tag from the Basic.Deliver frame
        """
        LOGGER.info("Acknowledging message %s", delivery_tag)
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):
        """
        Tells RabbitMQ that we would like to stop consuming by sending the
        Basic.Cancel RPC command.
        """
        if self._channel:
            LOGGER.info("Sending a Basic.Cancel RPC command to RabbitMQ")
            self._channel.basic_cancel(self.on_cancelok, self._consumer_tag)

    def on_cancelok(self, unused_frame):
        """
        Invoked by pika when RabbitMQ acknowledges the cancellation of a consumer.
        At this point we will close the channel. This will invoke the on_channel_closed
        method once the channel has been closed, which will in-turn close the connection.

        :param pika.frame.Method unused_frame: The Basic.CancelOk frame
        """
        LOGGER.info("RabbitMQ acknowledged the cancellation of the consumer")
        self.close_channel()

    def close_channel(self):
        """
        Closes the channel with RabbitMQ cleanly by issuing the
        Channel.Close RPC command.
        """
        LOGGER.info("Closing the channel")
        self._channel.close()

    def run(self):
        """
        Runs the consumer by connecting to RabbitMQ and then
        starting the IOLoop to block and allow the SelectConnection to operate.
        """
        self._connection = self.connect()
        self._connection.ioloop.start()

    def stop(self):
        """
        Cleanly shuts down the connection to RabbitMQ by stopping the consumer
        with RabbitMQ. When RabbitMQ confirms the cancellation, on_cancelok
        will be invoked by pika, which will then closing the channel and
        connection. The IOLoop is started again because this method is invoked
        when CTRL-C is pressed raising a KeyboardInterrupt exception. This
        exception stops the IOLoop which needs to be running for pika to
        communicate with RabbitMQ. All of the commands issued prior to starting
        the IOLoop will be buffered but not processed.
        """
        LOGGER.info("Stopping")
        self._closing = True
        self.stop_consuming()
        self._connection.ioloop.start()
        LOGGER.info("Stopped")

    def close_connection(self):
        """This method closes the connection to RabbitMQ."""
        LOGGER.info("Closing connection")
        self._connection.close()