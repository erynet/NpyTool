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

    def terminate(self):
        raise NotImplementedError


class FileSource(ISource):
    _is_loaded = False

    def __init__(self, path, recursive, output_file_name, worker_process_count, io_thread_per_worker, buffer_size, \
                 fnmatch_pattern, length_of_side, max_entry_count, augmentation_cmd, pre_process_cmd, logger):
        self._path = path
        self._recursive = recursive

        path, file_name = os.path.split(output_file_name)
        file_name_without_ext, _ = os.path.splitext(file_name)
        if path.__len__() == 0:
            file_name_with_path_but_ext = os.path.join(os.getcwdu(), file_name_without_ext)
        elif not os.path.exists(path):
            try:
                os.makedirs(path, mode=755)
            except Exception as e:
                file_name_with_path_but_ext = os.path.join(os.getcwdu(), file_name_without_ext)
            file_name_with_path_but_ext = os.path.join(path, file_name_without_ext)
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

        if io_thread_per_worker > 0:
            self._io_thread_per_worker = io_thread_per_worker
        else:
            self._io_thread_per_worker = 2

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

        # print "self.filtered.__len__()", self.filtered.__len__()
        # print "self._augmentation_power", self._augmentation_power
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

    def terminate(self):
        pass

    def export(self, compress=False):
        from lib import FileSourceWorker
        self._stand_by()
        for packed in self.filtered:
            self._in_q.put_nowait(packed)

        # for i in range(self._worker_process_count):
        #     self._producer_pool.append(multiprocessing.Process(target=self._producer_instance, args=(i,)))
        # self._consumer = multiprocessing.Process(target=self._consumer_instance, args=(compress, 3, 1))
        worker_kwars = {"root_path": self._path,
                        "file_name_with_path_but_ext": self._file_name_with_path_but_ext,
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

    def terminate(self):
        self._source.terminate()

    def export(self):
        self._source.export(self._compress)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="img2np")

    file_group = ap.add_argument_group("From File")
    file_group.add_argument("-p", "--path", type=str, default=None)
    file_group.add_argument("-f", "--fnmatch_pattern", type=str, default="*.*")
    file_group.add_argument("-r", "--recursive", action="store_true")
    file_group.add_argument("-s", "--length_of_side", type=int, default=64)

    ap.add_argument("-w", "--worker_process_count", type=int, default=0)
    ap.add_argument("-t", "--io_thread_per_worker", type=int, default=2)
    ap.add_argument("-b", "--buffer_size", type=int, default=65536)
    ap.add_argument("-c", "--compress", action="store_true")
    ap.add_argument("-o", "--outfile", type=str, default=None)
    ap.add_argument("-e", "--max_entry_count", type=int, default=100000000)
    ap.add_argument("-O", "--ordered", action="store_true")
    ap.add_argument("-a", "--augmentation", type=str, default="")
    ap.add_argument("-pp", "--preprocessing", type=str, default="")
    ap.add_argument("-l", "--logto", type=str, default="img2np.log")
    ap.add_argument("-L", "--loglevel", type=str, default="INFO")

    args = ap.parse_args()

    if not args.path:
        ap.print_help()
        sys.exit()

    try:

        with Log(args.logto, True, args.loglevel) as L:
            kwargs = {"path": args.path, \
                      "recursive": args.recursive, \
                      "worker_process_count": args.worker_process_count, \
                      "io_thread_per_worker": args.io_thread_per_worker, \
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
                kwargs["output_file_name"] = time.strftime("%Y%m%d-%H%M%S") + "-NpExport"

            N = NpyExport(source_type="File", source_kwargs=kwargs, compress=args.compress)

            N.export()
    except KeyboardInterrupt as e:
        # killing Signal
        pass
