import logging


def enforce_log_format():
    """
    Enables the the DSTIL logging format.
    """
    # Make the logger use our class.
    logging.setLoggerClass(DstilLogger)

    # Customise log level output for generic messages.
    logging.addLevelName(logging.CRITICAL, "FATAL")
    logging.addLevelName(logging.ERROR, "ERROR")
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.INFO, "INFO")
    logging.addLevelName(logging.DEBUG, "DEBUG")
    logging.addLevelName(logging.NOTSET, "TRACE")

    # Add EXPLAIN level for messages that contribute to "explainability"
    logging.addLevelName(DstilLogger.EXPLAIN, "EXPLAIN")

    # Configure log message format.
    logging.basicConfig(
        format='%(asctime)s.%(msecs)3d  %(levelname)7s 1 --- [%(threadName)15.15s] %(filename)36.36s:%(lineno)3d : %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S',
        level=logging.INFO
    )


class DstilLogger(logging.getLoggerClass()):
    """
    Logging class that augments the logging framework with an EXPLAIN log level.
    """

    # Our EXPLAIN level shouldn't ever be filtered out, so make it more critical than CRITICAL.
    EXPLAIN = logging.CRITICAL + 1

    def __init__(self, name, level=logging.INFO):
        """
        Initialises the logger with an identifying name and logging level.
        """
        # pylint: disable=useless-super-delegation
        super(DstilLogger, self).__init__(name, level)