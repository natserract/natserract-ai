import logging.config

log_config = {
    'logging_level': 'INFO',
    "logging": {
        "version": 1,
        "disable_existing_loggers": True,
        "root": {
            "level": "INFO",
        }
    }
}


def init_logger():
    try:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        logging.config.dictConfig(log_config['logging'])
        root_logger.setLevel(logging.getLevelName(log_config['logging_level']))
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f'Failed to load logging configuration. Using default configuration. Error: {e}')
