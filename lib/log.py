# -*- coding:utf-8 -*-

import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import multiprocessing, threading, logging, sys, traceback

try:
    import dill
except ImportError:
    import pip

    pip.main(["install", "dill==0.2.5"])
    import dill


class MultiProcessingHandler(logging.Handler):
    def __init__(self, filename, when, interval, backupcount):
        logging.Handler.__init__(self)

        self._handler = TimedRotatingFileHandler(filename=filename, when=when, interval=interval, \
                                                 backupCount=backupcount)
        self.queue = multiprocessing.Queue(-1)

        t = threading.Thread(target=self.receive)
        t.daemon = True
        t.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self._handler.close()
        logging.Handler.close(self)


class Log(object):
    def __init__(self, filepath=None, print_on_console=True, log_level="WARNING"):
        self.logger = logging.getLogger("npy_export")

        if log_level.upper() == u"DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif log_level.upper() == u"INFO":
            self.logger.setLevel(logging.INFO)
        elif log_level.upper() == u"WARNING":
            self.logger.setLevel(logging.WARNING)
        elif log_level.upper() == u"ERROR":
            self.logger.setLevel(logging.ERROR)
        elif log_level.upper() == u"CRITICAL":
            self.logger.setLevel(logging.CRITICAL)
        else:
            self.logger.setLevel(logging.WARNING)

        self._formatter = logging.Formatter("[%(levelname)s|%(filename)s] %(asctime)s > %(message)s")

        if filepath:
            path, file_name = os.path.split(filepath)
            if path.__len__() == 0:
                _output_file_name_with_path = os.path.join(os.getcwdu(), file_name)
            else:
                _output_file_name_with_path = filepath

            self._filehandler = MultiProcessingHandler(_output_file_name_with_path, when="D", interval=1, backupcount=7)
            self._filehandler.setFormatter(self._formatter)
            self.logger.addHandler(self._filehandler)

        if print_on_console:
            self._streamhandler = logging.StreamHandler()
            self._streamhandler.setFormatter(self._formatter)
            self.logger.addHandler(self._streamhandler)

    def d(self, msg):
        self.logger.debug(msg)

    def i(self, msg):
        self.logger.info(msg)

    def w(self, msg):
        self.logger.warning(msg)

    def e(self, msg):
        self.logger.error(msg)

    def c(self, msg):
        self.logger.critical(msg)

    def __enter__(self):
        self.d("Log.__enter__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.d("Log.__exit__")