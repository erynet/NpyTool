# -*- coding:utf-8 -*-

import sys, os
import multiprocessing

import imp

imp.reload(sys)
sys.setdefaultencoding("utf-8")

import time
import argparse

from lib import Log
from lib import Augmentation


class ISource(object):
    def export(self, compress=False):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class FileSource(ISource):
    _is_loaded = False

    def __init__(self, path, recursive, output_file_name, worker_process_count, buffer_size, \
                 fnmatch_pattern, length_of_side, max_entry_count, augmentation_cmd, pre_process_cmd, logger):
        self._path = path
        self._recursive = recursive

        path, file_name = os.path.split(output_file_name)
        file_name_without_ext, _ = os.path.splitext(file_name)
        if (path.__len__() == 0) or (not os.path.exists(path)):
            file_name_with_path_but_ext = os.path.join(os.getcwdu(), file_name_without_ext)
        else:
            file_name_with_path_but_ext = os.path.join(path, file_name_without_ext)
        self._file_name_with_path_but_ext = file_name_with_path_but_ext

        if worker_process_count > 0:
            self._worker_process_count = worker_process_count
        else:
            if pre_process_cmd.__len__() > 0:
                self._worker_process_count = int(multiprocessing.cpu_count() * 1.5)
            else:
                self._worker_process_count = int(multiprocessing.cpu_count() * 2)
        self._buffer_size = buffer_size

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

        self._in_q = multiprocessing.Queue(maxsize=-1)
        self._out_q = multiprocessing.Queue(maxsize=self._buffer_size)

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

    # def _init_instance(self):
    #     signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _stand_by(self):
        self._l.i("[FileSource] Stand by .... !!")

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

    def export(self, compress=False):
        from lib import FileSourceWorker
        self._stand_by()
        for packed in self.filtered:
            self._in_q.put_nowait(packed)

        # for i in range(self._worker_process_count):
        #     self._producer_pool.append(multiprocessing.Process(target=self._producer_instance, args=(i,)))
        # self._consumer = multiprocessing.Process(target=self._consumer_instance, args=(compress, 3, 1))
        worker_kwars = {"file_name_with_path_but_ext": self._file_name_with_path_but_ext,
                        "length_of_side": self._length_of_side,
                        "total_size": self._total_size,
                        "max_entry_count": self._max_entry_count,
                        "augmentation_cmd": self._augmentation_cmd,
                        "pre_process_cmd": self._pre_process_cmd,
                        "logger": self._l,
                        "in_q": self._in_q,
                        "out_q": self._out_q,
                        "instance_index": 0,
                        "compress": compress,
                        "role": 0}
        for i in range(self._worker_process_count):
            worker_kwars["instance_index"] = i
            self._producer_pool.append(multiprocessing.Process(target=FileSourceWorker, kwargs=worker_kwars))
        worker_kwars["in_q"] = None
        worker_kwars["role"] = 1
        self._consumer = multiprocessing.Process(target=FileSourceWorker, kwargs=worker_kwars)

        for p in self._producer_pool:
            p.start()
        self._consumer.start()

        self._clean_up()

    def close(self):
        pass


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
    file_group.add_argument("-r", "--recursive", action="store_true")
    file_group.add_argument("-s", "--length_of_side", type=int, default=64)

    # db_group = ap.add_argument_group("From Database")
    # db_group.add_argument("--mysqlhost", type=str, default="evancho.ery.wo.tc")
    # db_group.add_argument("--mysqlport", type=int, default=33007)
    # db_group.add_argument("--mysqluser", type=str, default="leatherdb_reader")
    # db_group.add_argument("--mysqlpasswd", type=str,
    #                       default="a8d48e84e1e4424272e39962330f4eacbb7302073b6d9562ede4127ed8a39b84")
    # db_group.add_argument("-i", "--index", type=int, default=0)

    ap.add_argument("-w", "--worker_process_count", type=int, default=0)
    ap.add_argument("-b", "--buffer_size", type=int, default=65536)
    ap.add_argument("-c", "--compress", action="store_true")
    ap.add_argument("-o", "--outfile", type=str, default=None)
    ap.add_argument("-e", "--max_entry_count", type=int, default=100000000)
    ap.add_argument("-O", "--ordered", action="store_true")
    ap.add_argument("-a", "--augmentation", type=str, default="")
    ap.add_argument("-pp", "--preprocessing", type=str, default="")
    ap.add_argument("-l", "--logto", type=str, default="npy_export.log")
    ap.add_argument("-L", "--loglevel", type=str, default="INFO")

    args = ap.parse_args()

    # if args.frompath:
    #     if not args.path:
    #         print "void or invalid path, use -P"
    #         sys.exit(0)
    #     print "%-24s: %s" % ("Operation Mode", "From Path",)
    #     print "%-24s: %s" % ("Target Path", args.path,)
    #     print "%-24s: %s" % ("Worker Process Count", args.worker_process_count,)
    #     print "%-24s: %s" % ("Filename Pattern", args.fnmatch_pattern,)
    #     print "%-24s: %s" % ("Length of Side", args.length_of_side,)
    # elif args.fromdb:
    #     pass
    #
    # print "%-24s: %s" % ("Compress", args.compress,)
    # print "%-24s: %s" % ("Max Entry Per File", args.max_entry_count,)
    # print "%-24s: %s" % ("Log File", args.logto,)
    # print "%-24s: %s" % ("Log Level", args.loglevel,)

    with Log(args.logto, True, args.loglevel) as L:
        if args.frompath:
            kwargs = {"path": args.path, \
                      "recursive": args.recursive, \
                      "worker_process_count": args.worker_process_count, \
                      "buffer_size": args.buffer_size, \
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
