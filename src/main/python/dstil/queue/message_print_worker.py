from .worker import Worker


class MessagePrintWorker(Worker):
    def __init__(self, rabbitmq_host, rabbitmq_port, rabbitmq_username, rabbitmq_password, queue, dead_letter_queue):
        super().__init__(rabbitmq_host, rabbitmq_port, rabbitmq_username, rabbitmq_password, queue, dead_letter_queue)

    def process_job(self, properties, body):
        self.logger.info("Properties: {}".format(properties))
        self.logger.info("Body: {}".format(body))
        return True

    def clean_up(self):
        pass
