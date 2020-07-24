import logging


class LoggerClass(object):
    def __init__(self, ):
        FORMAT = '%(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('defualt')

    def info(self, msg):
        self.logger.info(msg)


try:
    if Logger is not None:
        print('logger defined')
except NameError:
    Logger = LoggerClass()
