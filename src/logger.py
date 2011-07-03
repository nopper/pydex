import logging

logging.basicConfig()

def get_logger(name):
    log = logging.getLogger("stencil")
    log.setLevel(logging.INFO)
    return log
