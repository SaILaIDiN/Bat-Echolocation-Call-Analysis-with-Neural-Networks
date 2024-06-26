"""
    Define a function to create a logger class to log different features individually.
"""

import logging
from logging.handlers import RotatingFileHandler


class LoggerMaker:
    def __init__(self, loggingpurpose,
                 level=logging.INFO,
                 filehandling=False,
                 filelevel=logging.INFO,
                 fileformat="%(name)s:%(asctime)s:%(levelname)s:%(message)s",
                 streamhandling=False,
                 streamlevel=logging.INFO,
                 streamformat="%(name)s:%(asctime)s:%(levelname)s:%(message)s"
                 ):

        self.loggingpurpose = loggingpurpose
        self.level = level  # global level of logging statements

        self.filehandling = filehandling
        self.filehandler = None
        self.filelevel = filelevel
        self.fileformat = fileformat

        self.streamhandling = streamhandling
        self.streamhandler = None
        self.streamlevel = streamlevel
        self.streamformat = streamformat

        # create logger attached to class instance
        self.logger = logging.getLogger(self.loggingpurpose)
        self.logger.setLevel(self.level)

        if self.filehandling:
            self.attach_filehandler()
        if self.streamhandling:
            self.attach_streamhandler()

    def attach_filehandler(self):
        self.filehandler = RotatingFileHandler(f"{self.loggingpurpose}.log")
        self.filehandler.setLevel(self.filelevel)
        self.filehandler.setFormatter(logging.Formatter(self.fileformat))
        # Add filehandler to logger
        self.logger.addHandler(self.filehandler)

    def attach_streamhandler(self):
        self.streamhandler = logging.StreamHandler()
        self.streamhandler.setLevel(self.streamlevel)
        self.streamhandler.setFormatter(logging.Formatter(self.streamformat))
        # Add streamhandler to logger
        self.logger.addHandler(self.streamhandler)
