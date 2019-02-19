#!/usr/bin/env python
import argparse
import json
import logging
import os
import pika
import shutil
import signal
import sys
import time
from abc import ABC, abstractmethod
from dstil.logging import logger

class Worker(ABC):
    def __init__(self, rabbitmq_host, rabbitmq_port, rabbitmq_username, rabbitmq_password, queue_name, dead_letter_queue_name):
        self.logger = logging.getLogger(__name__)

        # Queue connection parameters.
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_username = rabbitmq_username
        self.rabbitmq_password = rabbitmq_password
        self.queue_name = queue_name
        self.dead_letter_queue_name = dead_letter_queue_name
        self.connection = None

        # Exit behaviour.
        signal.signal(signal.SIGTERM, self.handle_sigterm)
        self.exiting = False

    def run(self):
        """
        Listens forever to the configured queue and calls process_job() for each message received.
        Reconnects to the queue if the connection is dropped.
        """
        while not self.exiting:
            try:
                # Create RabbitMQ connection.
                credentials = pika.PlainCredentials(self.rabbitmq_username, self.rabbitmq_password)
                self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.rabbitmq_host, port=self.rabbitmq_port, credentials=credentials))
                channel = self.connection.channel()

                # Declare a dead-letter exchange and queue for failed messages.
                if self.dead_letter_queue_name is not None:
                    channel.exchange_declare(exchange=self.dead_letter_queue_name, exchange_type="direct", durable=True)
                    channel.queue_declare(queue=self.dead_letter_queue_name, durable=True)
                    channel.queue_bind(exchange=self.dead_letter_queue_name,
                                       queue=self.dead_letter_queue_name)

                    # Declare a durable queue that will persist across broker restarts,
                    # with QOS to only dispatch jobs after the previous one is finished.
                    channel.queue_declare(queue=self.queue_name,
                                          durable=True,
                                          arguments={
                                              "x-dead-letter-exchange" : self.dead_letter_queue_name,
                                              "x-dead-letter-routing-key" : self.dead_letter_queue_name
                                          })
                else:
                    channel.queue_declare(queue=self.queue_name, durable=True)

                # Set prefetch count to only fetch one message at a time.
                channel.basic_qos(prefetch_count=1)

                # Start consuming.
                self.logger.info("Waiting for jobs")
                channel.basic_consume(self.__process_job, queue=self.queue_name)
                channel.start_consuming()
            except KeyboardInterrupt as e:
                self.__clean_up()
            except SystemExit as e:
                self.__clean_up()
            except Exception as e:
                if not self.exiting:
                    self.logger.exception(e)
                    self.logger.error("Cannot reach queue; trying again in 1 sec...")
                    time.sleep(1)

    def handle_sigterm(self, signum, frame):
        """
        Catches the SIGTERM signal and cleans up resources gracefully.
        """
        self.__clean_up()

    def __clean_up(self):
        """
        Cleans up any resources/connections in use before exiting.
        """
        self.exiting = True
        self.clean_up()  # Subclass hook.
        if self.connection is not None:
            self.connection.close()

    def __process_job(self, ch, method, properties, body):
        """
        Handles incoming messages (absolute paths to image files) coming in from the queue.
        Passes each path through the pipeline and delegates handling of the resultant encoding/error/warnings.
        """
        success = None

        # Call the subclass hook to process the message.
        try:
            success = self.process_job(properties, body)
        except Exception as e:
            success = False
            self.logger.exception(e)

        assert isinstance(success, bool), "process_job() must return a boolean value"

        # Send acknowledgement to the broker if the job succeeded, otherwise send a rejection message.
        if success:
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            self.logger.info("Handler returned False; rejecting message.")
            if self.dead_letter_queue_name is None:
                self.logger.info("Note: There is no dead letter queue configured, so this message will be dropped.")

            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

    @abstractmethod
    def process_job(self, properties, body):
        """
        Subclass hook for processing incoming jobs. This method must return True or False.
        """
        pass

    @abstractmethod
    def clean_up(self):
        """
        Subclass hook for performing any necessary cleanup.
        """
        pass
