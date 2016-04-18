# -*- coding:utf-8 -*-

import sys, os
# import threading
# from multiprocessing import Process, Queue
import signal
import multiprocessing

import imp

imp.reload(sys)
sys.setdefaultencoding("utf-8")

import time
from datetime import datetime, timedelta
import re, zlib, bz2, base64
from struct import pack, unpack
import copy

try:
    import numpy as np
except ImportError:
    import pip

    pip.main(["install", "numpy"])
    import numpy as np

try:
    import mysql.connector as mc
except ImportError:
    import pip

    pip.main(["install", "mysql-connector-python-rf==2.1.3"])
    import mysql.connector as mc

try:
    from PIL import Image
except ImportError:
    import pip

    pip.main(["install", "Pillow==3.2.0"])
    from PIL import Image

try:
    import ujson as json
except ImportError:
    import pip

    pip.main(["install", "ujson==1.35"])
    import ujson as json

import argparse

import Queue

# from Queue import Queue, Empty

import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import multiprocessing, threading, logging, sys, traceback


class MultiProcessingHandler(logging.Handler):
    def __init__(self, filename, when, interval, backupcount):
        logging.Handler.__init__(self)

        self._handler = TimedRotatingFileHandler(filename=filename, when=when, interval=interval,
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
        # print "MultiProcessingHandler.receive"

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

            # self._filehandler = logging.handlers.\
            # TimedRotatingFileHandler(_output_file_name_with_path, when="D", interval=1, backupCount=7)
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


class Augmentation(object):
    def __init__(self, cmd, logger=None):
        self._cmd = cmd.lower()
        self._l = logger

        self._regex = re.compile("([ftr]{1})")
        self._chain = []

        self._parse()

    def log(self, msg):
        if not self._l:
            return
        self._l.i("[Augmentation] " + msg)

    def _parse(self):
        if self._cmd.__len__() == 0:
            return
        tokens = self._regex.findall(self._cmd)
        if tokens.__len__() == 0:
            return

        if "f" in tokens:
            # flip
            self._chain.append(("_flip", "Flip", 2))
        if "t" in tokens:
            # transpose
            self._chain.append(("_transpose", "Transpose", 2))
        if "r" in tokens:
            # rotation
            self._chain.append(("_rotate", "Rotate", 4))

    @staticmethod
    def calc_power(cmd):
        tokens = re.findall("([ftr]{1})", cmd)
        p = 1
        if tokens.__len__() == 0:
            return p

        if "f" in tokens:
            # flip
            p *= 2
        if "t" in tokens:
            # transpose
            p *= 2
        if "r" in tokens:
            # rotation
            p *= 4
        return p

    def explain(self):
        if self._chain.__len__() == 0:
            return None, 1
        desc_list = []
        p = 1
        for _, desc, power in self._chain:
            p *= power
            desc_list.append(desc)
        return " >> ".join(desc_list) + " (x%d)" % p, p

    def go(self, mat):
        _mats = {"O": mat}
        if self._chain.__len__() == 0:
            return _mats
        for func, _, _ in self._chain:
            _mats.update(self.__getattribute__(func)(_mats))
        return _mats

    @staticmethod
    def _flip(mats):
        new_mats = {}
        for k in mats:
            new_mats[k + "F"] = np.fliplr(mats[k])
        return new_mats

    @staticmethod
    def _transpose(mats):
        new_mats = {}
        for k in mats:
            new_mats[k + "T"] = np.transpose(mats[k])
        return new_mats

    @staticmethod
    def _rotate(mats):
        new_mats = {}
        for k in mats:
            for c in range(1, 4):
                new_mats[k + "R%03d" % (90 * c)] = np.rot90(mats[k], c)
        return new_mats


class PreProcess(object):
    def __init__(self, cmd, logger=None):
        self._cmd = cmd
        self._l = logger

        self._regex = re.compile("([\w\d\.\-]+)")
        self._chain = []

        self._parse()

    def log(self, msg):
        if not self._l:
            return
        self._l.i("[PreProcess] " + msg)

    def _parse(self):
        if self._cmd.__len__() == 0:
            return
        seps = self._cmd.split(",")
        for sep in seps:
            tokens = self._regex.findall(sep)

            try:
                func = "_" + tokens[0]
                f = self.__getattribute__(func)
                argc = f.func_code.co_argcount
                # 아규먼트가 1이면, 토큰은 2 이상이어야 한다.
                if tokens.__len__() < argc:
                    # 인자가 부족하다.
                    Log("Not enough arguments, Token : %s" % (sep,))
                    continue
                args = tokens[1:argc]
                desc = "func=%s, args=%s" % (func, str(args))
                self._chain.append((func, args, desc))
            except AttributeError:
                Log("No such function, Token : %s" % (sep,))
                continue

    def explain(self):
        if self._chain.__len__() == 0:
            return None
        desc_list = []
        for _, _, desc in self._chain:
            desc_list.append(desc)
        return " >> ".join(desc_list)

    def go(self, mat):
        if self._chain.__len__() == 0:
            return mat

        _mat = mat
        for func, args, _ in self._chain:
            _mat = self.__getattribute__(func)(_mat, *args)
        return _mat

    @staticmethod
    def _zcaw(mat, epsilon="0.1"):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape
        epsilon = float(epsilon)
        sigma = np.dot(mat, mat.T) / mat.shape[1]
        U, S, _ = np.linalg.svd(sigma)
        zca_mat = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)
        return np.dot(zca_mat, mat).reshape((shape[0], shape[1]))

    @staticmethod
    def _pcaw(mat):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape
        U, _, V = np.linalg.svd(mat)
        return np.dot(U, V).reshape((shape[0], shape[1]))

    @staticmethod
    def _gaussian_noise(mat, p="0.5"):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape

        p = float(p)
        n = 1
        s = np.random.binomial(n, p, shape[0] * shape[1])
        noised_mat = mat.reshape(shape[0] * shape[1])

        for i, out in enumerate(s):
            if out == 1:
                noised_mat[i] = 0

        return noised_mat.reshape((shape[0], shape[1]))

    @staticmethod
    def _histeq(mat):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape

        histogram = np.zeros(256)
        intensity = np.zeros(256)
        for r in range(shape[0]):
            for c in range(shape[1]):
                histogram[mat[r, c]] += 1.0
        for i in range(256):
            histogram[i] /= shape[0] * shape[1]
        intensity[0] = histogram[0]
        for i in range(1, 256):
            intensity[i] = intensity[i - 1] + histogram[i]
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        im = mat.reshape((shape[0] * shape[1]))
        for i in range(256):
            intensity[i] *= 255
        for i in range(shape[0] * shape[1]):
            img[i] = int(intensity[im[i]])

        return img.reshape((shape[0], shape[1]))


class ISource(object):
    def export(self, compress=False):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class FileSource(ISource):
    _is_loaded = False

    def __init__(self, path, recursive, output_file_name, worker_process_count, fnmatch_pattern, length_of_side, \
                 max_entry_count, augmentation_cmd, pre_process_cmd, logger=Log()):
        self._path = path
        self._recursive = recursive

        path, file_name = os.path.split(output_file_name)
        if path.__len__() == 0:
            _output_file_name_with_path_and_ext = os.path.join(os.getcwdu(), file_name)
        elif not os.path.exists(path):
            _output_file_name_with_path_and_ext = os.path.join(os.getcwdu(), file_name)
        file_name_with_path_but_ext, _ = os.path.splitext(_output_file_name_with_path_and_ext)
        self._file_name_with_path_but_ext = file_name_with_path_but_ext

        if worker_process_count > 0:
            self._worker_process_count = worker_process_count
        else:
            if pre_process_cmd.__len__() > 0:
                self._worker_process_count = int(multiprocessing.cpu_count() * 1.5)
            else:
                self._worker_process_count = int(multiprocessing.cpu_count() * 2)

        self._fnmatch_pattern = fnmatch_pattern
        self._length_of_side = length_of_side
        self._max_entry_count = max_entry_count

        self._augmentation_cmd = augmentation_cmd
        self._augmentation_power = Augmentation.calc_power(augmentation_cmd)
        self._pre_process_cmd = pre_process_cmd

        self._l = logger

        self._total_size = 0

        self._current_index = 0

        self._ndarr = None

        # 쓰레드풀 대기

        self._in_q = multiprocessing.Queue(maxsize=0)
        self._out_q = multiprocessing.Queue(maxsize=16384)

        self._suicide = False

        self._producer_pool = []
        self._consumer = None

    def print_summary(self):
        pass

    def _recursive_collect(self):
        import fnmatch
        path_stack = [self._path]
        target_bag = {}
        while path_stack:
            path_to_go = path_stack.pop()
            g = os.walk(path_to_go)
            try:
                root, dirs, files = g.next()
            except Exception:
                continue
            matched = fnmatch.filter(files, self._fnmatch_pattern)
            if matched:
                if root not in target_bag:
                    target_bag[root] = []
                target_bag[root] += matched
            if self._recursive:
                for dir in dirs:
                    path_stack.append(os.path.join(root, dir))
        return target_bag

    def _init_instance(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _stand_by(self):
        self._l.i("[FileSource] Stand by ....")

        collection = self._recursive_collect()
        self.filtered = []
        for path in collection:
            for file in collection[path]:
                self.filtered.append((path, file))
        self.filtered.sort()

        self._total_size = self.filtered.__len__() * self._augmentation_power
        self._l.i("[FileSource] I found %d files to work in %s" % (self.filtered.__len__(), self._path))

    def _clean_up(self):
        self._consumer.join()
        self._l.i("[FileSource] Cleaning up ....")
        self._l.i("[FileSource] Waiting for queues ...")
        self._in_q.close()
        self._in_q.join_thread()
        self._out_q.close()
        self._out_q.join_thread()
        self._l.i("[FileSource] Cleaning Ok")

    def _producer_instance(self, instance_index):
        idx = instance_index
        side = self._length_of_side

        count = 0

        au = Augmentation(cmd=self._augmentation_cmd, logger=self._l)
        au_desc, power = au.explain()
        pp = PreProcess(cmd=self._pre_process_cmd, logger=self._l)
        pp_desc = pp.explain()

        init_msg = "[Producer #%2d] Init (pid: %5d)" % (idx, os.getpid())
        if au_desc:
            init_msg += ", Augmentaion : %s" % (au_desc,)
        if pp_desc:
            init_msg += ", PreProcessing : %s" % (pp_desc,)
        self._l.i(init_msg)

        while True:
            try:
                path, fn = self._in_q.get(block=True, timeout=1.5)
            except Queue.Empty:
                # 초반에 읽어야 할 파일 목록을 다 부어버리기 떄문에, 이 예외로 빠진다는건
                # 처리할 물량이 더이상 없다는 듯이다. 고로 종료한다.
                break

            _start = time.time()

            results = []
            full_path = os.path.join(path, fn)
            try:
                img = Image.open(full_path)

                if img.size[0] != side or img.size[1] != side:
                    self._l.w("[Producer #%2d] Anomaly : %s" % (idx, full_path))
                    continue

                mat = np.asarray(img)

                # augmentation 구동
                mats = au.go(mat)
                for aug in mats:
                    # 전처리 _chain 구동
                    results.append((aug, pp.go(mats[aug])))
            except IOError, e:
                self._l.e("[Producer #%2d] IOError, %s, %s" % (idx, full_path, e.message))
            except Exception, e:
                self._l.e("[Producer #%2d] Exception, %s, %s" % (idx, full_path, e.message))
            finally:
                if "img" in __name__ and img:
                    img.close()

            dt = int((time.time() - _start) * 1000 / power)
            self._l.d("[Producer #%2d] %s, processed" % (idx, full_path))
            for result in results:
                self._out_q.put((path, fn, result[0], dt, result[1]))
            count += 1
        self._l.i("[Producer #%2d] Exit, Processed Count %d, Exported Count %d" % (idx, count, count * power))

    def _consumer_instance(self, compress=False, wait_sec_for_producer=5, timeout_sec=3):
        def _progress():
            mean = float(np.mean(_st))
            progress = (total_count * 100) / total_entry
            dt = time.time() - _start
            estimation = (dt / total_count) * total_entry
            self._l.i("[Consumer] [%3d%%] %.5f ms per image, %.2f/%.2f sec" % (progress, mean, dt, estimation))

        def _dump():
            if compress:
                self._l.i("[Consumer] Compress start ...")
                if max_entry <= total_entry:
                    np_fn = filename_base + "_%03d" % rotate + ".npz"
                else:
                    np_fn = filename_base + ".npz"
                _compress_start = time.time()
                np.savez_compressed(np_fn, arr=_arr, cat=_cat)
                self._l.i("[Consumer] Compress end (%.2fs) -> %s" % (time.time() - _compress_start, np_fn))
            else:
                if max_entry <= total_entry:
                    np_fn = filename_base + "_%03d" % rotate + ".npy"
                    cat_fn = filename_base + "_%03d" % rotate + ".cat"
                else:
                    np_fn = filename_base + ".npy"
                    cat_fn = filename_base + ".cat"
                _dump_start = time.time()
                np.save(np_fn, _arr)
                self._l.i("[Consumer] Dump done (%.2fs) -> %s" % (time.time() - _dump_start, np_fn))
                with open(cat_fn, "w") as fp:
                    fp.write(json.dumps(_cat))
                    fp.flush()
                self._l.i("[Consumer] Catalog generated -> %s" % (cat_fn,))

        count = 0
        total_count = 0
        rotate = 0

        total_entry = self._total_size
        max_entry = self._max_entry_count
        length_of_side = self._length_of_side
        filename_base = self._file_name_with_path_but_ext

        self._l.i("[Consumer] Init (pid: %5d), timeout is %d sec" % (os.getpid(), timeout_sec))
        self._l.i("[Consumer] Wait for producer %d sec" % (wait_sec_for_producer,))
        for i in range(wait_sec_for_producer):
            self._l.i("[Consumer] %d ..." % (i + 1,))
            time.sleep(1)
        self._l.i("[Consumer] Go")

        _cat = {"parameter": {"augmentation:": self._augmentation_cmd, "preprocessing": self._pre_process_cmd}, \
                "catalog": {}}
        _st = []

        if max_entry > total_entry:
            _arr = np.zeros((total_entry, length_of_side, length_of_side))
        else:
            _arr = np.zeros((max_entry, length_of_side, length_of_side))

        _start = time.time()
        while True:
            try:
                path, fn, aug, dt, mat = self._out_q.get(block=True, timeout=timeout_sec)
            except Queue.Empty:
                # 이건 끝났다고 봐도 된다.
                # 고로 여기서 break 하고
                _progress()
                _dump()
                break

            _cat["catalog"][count] = (path, fn, aug)
            _st.append(dt)
            _arr[count] = mat

            count += 1
            total_count += 1

            if total_count > 0 and (total_count % ((total_entry / 100) * 3)) == 0:
                _progress()

            if count > 0 and ((count % max_entry) == (max_entry - 1)):
                _dump()

                count = 0
                rotate += 1

                _cat = {"parameter": {"augmentation:": self._augmentation_cmd, "preprocessing": self._pre_process_cmd}, \
                        "catalog": {}}
                if total_entry - total_count >= max_entry:
                    _arr = np.zeros((max_entry, length_of_side, length_of_side))
                else:
                    _arr = np.zeros((total_entry - total_count, length_of_side, length_of_side))

        self._l.i("[Consumer] total %d elements processed" % (total_count,))
        self._l.i("[Consumer] Exit")

    def export(self, compress=False):
        self._stand_by()
        # 여기에서
        # count = 0
        for packed in self.filtered:
            # count += 1
            self._in_q.put_nowait(packed)
        # print "#####", count

        for i in range(self._worker_process_count):
            self._producer_pool.append(multiprocessing.Process(target=self._producer_instance, args=(i,)))
        self._consumer = multiprocessing.Process(target=self._consumer_instance, args=(compress, 3, 1))

        for p in self._producer_pool:
            p.start()
        self._consumer.start()

        self._clean_up()

    def close(self):
        pass
        # self._suicide = True


class DatabaseSource(ISource):
    def __init__(self, mysqlhost, mysqlport, mysqluser, mysqlpasswd, target_index):
        pass

    def _load(self):
        raise NotImplementedError

    def export(self, compress):
        raise NotImplementedError

    def close(self):
        pass


class NpyExport(object):
    _source = None

    def __init__(self, source_type, source_kwargs, compress=False):
        if source_type == "File":
            self._source = FileSource(**source_kwargs)
        elif source_type == "Database":
            self._source = DatabaseSource(**source_kwargs)

        self._compress = compress

    def export(self):
        self._source.export(self._compress)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="npy_export")

    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("-p", "--frompath", action="store_true")
    source_group.add_argument("-d", "--fromdb", action="store_true")

    file_group = ap.add_argument_group("From File")
    file_group.add_argument("-P", "--path", type=str, default=None)
    file_group.add_argument("-f", "--fnmatch_pattern", type=str, default="*.*")
    file_group.add_argument("-R", "--recursive", action="store_true")
    file_group.add_argument("-s", "--length_of_side", type=int, default=64)

    db_group = ap.add_argument_group("From Database")
    db_group.add_argument("--mysqlhost", type=str, default="evancho.ery.wo.tc")
    db_group.add_argument("--mysqlport", type=int, default=33007)
    db_group.add_argument("--mysqluser", type=str, default="leatherdb_reader")
    db_group.add_argument("--mysqlpasswd", type=str,
                          default="a8d48e84e1e4424272e39962330f4eacbb7302073b6d9562ede4127ed8a39b84")
    db_group.add_argument("-i", "--index", type=int, default=0)

    ap.add_argument("-w", "--worker_process_count", type=int, default=0)
    ap.add_argument("-c", "--compress", action="store_true")
    ap.add_argument("-o", "--outfile", type=str, default=None)
    ap.add_argument("-e", "--max_entry_count", type=int, default=100000000)
    ap.add_argument("-a", "--augmentation", type=str, default="")
    ap.add_argument("-pp", "--preprocessing", type=str, default="")
    ap.add_argument("-l", "--logto", type=str, default="npy_export.log")
    ap.add_argument("-L", "--loglevel", type=str, default="INFO")

    args = ap.parse_args()

    if args.frompath:
        if not args.path:
            print "void or invalid path, use -P"
            sys.exit(0)
        print "%-24s: %s" % ("Operation Mode", "From Path",)
        print "%-24s: %s" % ("Target Path", args.path,)
        print "%-24s: %s" % ("Worker Process Count", args.worker_process_count,)
        print "%-24s: %s" % ("Filename Pattern", args.fnmatch_pattern,)
        print "%-24s: %s" % ("Length of Side", args.length_of_side,)
    elif args.fromdb:
        pass

    print "%-24s: %s" % ("Compress", args.compress,)
    print "%-24s: %s" % ("Max Entry Per File", args.max_entry_count,)
    print "%-24s: %s" % ("Log File", args.logto,)
    print "%-24s: %s" % ("Log Level", args.loglevel,)

    with Log(args.logto, False, args.loglevel) as L:

        if args.frompath:
            kwargs = {"path": args.path, \
                      "recursive": args.recursive, \
                      "worker_process_count": args.worker_process_count, \
                      "fnmatch_pattern": args.fnmatch_pattern, \
                      "length_of_side": args.length_of_side, \
                      "max_entry_count": args.max_entry_count, \
                      "augmentation_cmd": args.augmentation, \
                      "pre_process_cmd": args.preprocessing, \
                      "logger": L}
            if args.outfile:
                kwargs["output_file_name"] = args.outfile
            else:
                kwargs["output_file_name"] = time.strftime("%Y%m%d-%H%M%S") + "-NpyExport"

            N = NpyExport(source_type="File", source_kwargs=kwargs, compress=args.compress)
        elif args.fromdb:
            N = None
            pass

        N.export()
