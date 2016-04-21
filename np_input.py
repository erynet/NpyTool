# -*- coding:utf-8 -*-

import sys, os
import imp

imp.reload(sys)
sys.setdefaultencoding("utf-8")

import gc

import numpy as np

try:
    import ujson as json
except ImportError:
    import pip

    pip.main(["install", "ujson==1.35"])
    import ujson as json


class NpArrayLoader(object):
    def __init__(self, catalog):
        self._catalog = catalog
        # print "CATALOG", catalog

        if not os.path.exists(catalog):
            raise IOError("There are no such file")
        try:
            with open(catalog, "rb") as fp:
                cat = json.loads(fp.read())
        except IOError, e:
            print catalog
            print str(e)
            raise
        except Exception, e:
            print str(e)
            raise

        source_path, _ = os.path.split(catalog)

        if cat["compress"]:
            self._core = NpzArray(cat, source_path)
        else:
            self._core = NpyArray(cat, source_path)

    def get_metadata(self, idx):
        return self._core.get_metadata(idx)

    def get(self, idx):
        return self._core.get(idx)

    def get_batch(self, size=1000, pos=0):
        return self._core.get_batch(size=size, pos=pos)


class INpSource(object):
    def __init__(self, catalog_dict, source_path):
        self.cat = catalog_dict
        self.source_path = source_path

        self.total_count = self.cat["total_count"]
        self.parameter = self.cat["parameter"]
        self.dict_augmentation_flags = self.cat["dict_augmentation_flags"]
        self.dict_filenames = self.cat["dict_filenames"]
        self.dict_src_paths = self.cat["dict_src_paths"]
        self.np_files = self.cat["np_files"]
        self.ranges = self.cat["ranges"]
        self.catalog = self.cat["catalog"]

        self.entry_per_file = self.cat["entry_per_file"]

        self._opened_np_file_idx_set = set()
        self._opened_np_file_data = {}

    def _unload_np_files(self, idle_set=None, busy_set=None):
        raise NotImplementedError

    def _load_np_files(self, busy_set):
        raise NotImplementedError

    def _calc_range(self, pos, size):
        start = (self.catalog[pos][0], pos % self.entry_per_file)
        if (pos + size - 1) < self.total_count:
            end = (self.catalog[pos + size - 1][0], (pos + size - 1) % self.entry_per_file)
            count = size
        else:
            end = (self.catalog[self.total_count - 1][0], (self.total_count - 1) % self.entry_per_file)
            count = self.total_count - pos
        return start, end, count

    def _synth_ndarray_by_range(self, start, end):
        busy_set = set(range(start[0], end[0] + 1))
        self._unload_np_files(busy_set=busy_set)
        self._load_np_files(busy_set=busy_set)
        middle = []
        for idx in range(start[0] + 1, end[0]):
            middle.append(self._opened_np_file_data[idx])
        if start[0] == end[0]:
            return self._opened_np_file_data[start[0]][start[1]:end[1] + 1]
        else:
            return np.concatenate([self._opened_np_file_data[start[0]][start[1]:]] + middle + \
                                  [self._opened_np_file_data[end[0]][:end[1] + 1]])

    def get_metadata(self, idx):
        return {"src_path": self.dict_src_paths[self.catalog[idx][1]],
                "src_filename": self.dict_filenames[self.catalog[idx][2]],
                "np_file": self.np_files[self.catalog[idx][0]],
                "parameter": self.parameter,
                "augmentation_flag": self.dict_augmentation_flags[self.catalog[idx][3]]}

    def get(self, idx):
        np_file_idx = self.catalog[idx][0]
        if np_file_idx not in self._opened_np_file_idx_set:
            self._load_np_files({np_file_idx})
        return self._opened_np_file_data[np_file_idx][idx % self.entry_per_file]

    def get_ranged(self, size=0, pos=0):
        if size == 0:
            # read full range
            start, end, _ = self._calc_range(pos, self.total_count)
        else:
            start, end, _ = self._calc_range(pos, size)
        return self._synth_ndarray_by_range(start, end)

    def get_batch(self, size=1000, pos=0):
        current = pos
        while True:
            start, end, count = self._calc_range(current, size)
            yield self._synth_ndarray_by_range(start, end)
            current += size
            if count < size:
                break


class NpyArray(INpSource):
    def _unload_np_files(self, idle_set=None, busy_set=None):
        if idle_set:
            for idx in idle_set:
                if idx in self._opened_np_file_data:
                    del self._opened_np_file_data[idx]
            gc.collect()
            self._opened_np_file_idx_set.difference_update(idle_set)
        if busy_set:
            # print "UnLoad : ", busy_set, self._opened_np_file_idx_set.difference(busy_set)
            for idx in self._opened_np_file_idx_set.difference(busy_set):
                if idx in self._opened_np_file_data:
                    del self._opened_np_file_data[idx]
            gc.collect()
            self._opened_np_file_idx_set.intersection_update(busy_set)

    def _load_np_files(self, busy_set):
        # print "Load : ", busy_set, busy_set.difference(self._opened_np_file_idx_set)
        for idx in busy_set.difference(self._opened_np_file_idx_set):
            print self.source_path
            print self.np_files[idx].decode("utf8")
            self._opened_np_file_data[idx] = np.load(os.path.join(self.source_path, self.np_files[idx]), "r")
        self._opened_np_file_idx_set.update(busy_set)


class NpzArray(INpSource):
    def _unload_np_files(self, idle_set=None, busy_set=None):
        if idle_set:
            for idx in idle_set:
                if idx in self._opened_np_file_data:
                    del self._opened_np_file_data[idx]
            gc.collect()
            self._opened_np_file_idx_set.difference_update(idle_set)
        if busy_set:
            # print "UnLoad : ", busy_set, self._opened_np_file_idx_set.difference(busy_set)
            for idx in self._opened_np_file_idx_set.difference(busy_set):
                if idx in self._opened_np_file_data:
                    del self._opened_np_file_data[idx]
            gc.collect()
            self._opened_np_file_idx_set.intersection_update(busy_set)

    def _load_np_files(self, busy_set):
        # print "Load : ", busy_set, busy_set.difference(self._opened_np_file_idx_set)
        for idx in busy_set.difference(self._opened_np_file_idx_set):
            self._opened_np_file_data[idx] = np.load(os.path.join(self.source_path, self.np_files[idx]), "r")["arr"]
        self._opened_np_file_idx_set.update(busy_set)

if __name__ == "__main__":
    import time
    _start = time.time()
    NAL = NpArrayLoader(u"g:\\test14.cat")
    a = 0
    for b in NAL.get_batch(size=750, pos=0):
        a += b.__len__()
    print a

    print "delta : %.3f" % (time.time() - _start,)
