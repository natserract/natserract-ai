import logging.config

log_config = {
    'logging_level': 'INFO',
    "logging": {
        "version": 1,
        "disable_existing_loggers": True,
    }
}


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'


class ColoredFormatter(logging.Formatter):
    FORMAT = "%(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    COLORS = {
        'WARNING': Colors.YELLOW,
        'INFO': Colors.GREEN,
        'DEBUG': Colors.BLUE,
        'CRITICAL': Colors.RED,
        'ERROR': Colors.MAGENTA
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, Colors.WHITE)
        formatter = logging.Formatter(color + self.FORMAT + Colors.RESET)
        return formatter.format(record)


def init_logger():
    try:
        root_logger = logging.getLogger()

        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(handler)

        logging.config.dictConfig(log_config['logging'])
        root_logger.setLevel(logging.getLevelName(log_config['logging_level']))
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f'Failed to load logging configuration. Using default configuration. Error: {e}')
