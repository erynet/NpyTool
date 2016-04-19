# -*- coding:utf-8 -*-

import sys, os
import imp

imp.reload(sys)
sys.setdefaultencoding("utf-8")

import gc

import numpy as np


class ISourceLoader(object):
    pass


class NpyArrayLoader(ISourceLoader):
    def __init__(self, catalog):
        self._catalog = catalog
        # print "CATALOG", catalog

        if not os.path.exists(self._catalog):
            raise IOError("There are no such file")
        try:
            import json
            with open(self._catalog, "rb") as fp:
                self.cat = json.loads(fp.read())
        except IOError, e:
            print self._catalog
            print str(e)
            raise
        except Exception, e:
            print str(e)
            raise

        self.source_path, _ = os.path.split(self._catalog)

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

    #     if self.cat.ranges.__len__() > 128:
    #         self._find_np_file = self._find_np_file_by_binsearch
    #     else:
    #         self._find_np_file = self._find_np_file_by_seq
    #
    # def _find_np_file_by_binsearch(self, idx):
    #     b = 0
    #     t = self.ranges.__len__() - 1
    #     c = 0
    #
    #     for _ in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
    #         c = (b + t) / 2
    #         if abs(self.ranges[b][0] - idx) + abs(self.ranges[b][1] - idx) >= \
    #                         abs(self.ranges[t][0] - idx) + abs(self.ranges[t][1] - idx):
    #             # 바닥쪽의 편차의 절대값이 더 크다. -> b = c + 1
    #             b = c + 1
    #         else:
    #             # 바닥쪽의 편차의 절대값이 더 크다. -> t = c
    #             t = c
    #         if t - b < 3:
    #             break
    #     for i in (b, b + 1):
    #         if self.ranges[i][0] <= idx <= self.ranges[i][1]:
    #             return i
    #     raise ValueError("Something is Wrong")
    #
    # def _find_np_file_by_seq(self, idx):
    #     if idx > (self.total_count / 2):
    #         # desc
    #         for i in xrange(self.total_count - 1, -1, -1):
    #             if self.ranges[i][0] <= idx <= self.ranges[i][1]:
    #                 return self.np_files[i]
    #     else:
    #         # asc
    #         for i in xrange(self.total_count):
    #             if self.ranges[i][0] <= idx <= self.ranges[i][1]:
    #                 return self.np_files[i]
    #     raise ValueError("Something is Wrong")

    def _unload_np_files(self, idle_set=None, busy_set=None):
        # idle_set 과 busy_set 셋중 하나만 들어오면 맞는 동작을 한다.
        if idle_set:
            # 노는 set 을 직접 제거
            for idx in idle_set:
                if idx in self._opened_np_file_data:
                    self._opened_np_file_data[idx]._mmap.close()
                    del self._opened_np_file_data[idx]
            gc.collect()
            self._opened_np_file_idx_set.difference_update(idle_set)
        if busy_set:
            # busy_set 을 제외한 집합을 전부 제거
            for idx in self._opened_np_file_idx_set.difference(busy_set):
                if idx in self._opened_np_file_data:
                    self._opened_np_file_data[idx]._mmap.close()
                    del self._opened_np_file_data[idx]
            gc.collect()
            self._opened_np_file_idx_set.intersection_update(busy_set)

    def _load_np_files(self, busy_set):
        for idx in busy_set.difference(self._opened_np_file_idx_set):
            self._opened_np_file_data[idx] = np.load(os.path.join(self.source_path, self.np_files[idx]), "r")
        self._opened_np_file_idx_set.update(busy_set)

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
        self._load_np_files(busy_set)
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
                # "range": self.ranges[self.catalog[idx][0]],
                "parameter": self.parameter,
                "augmentation_flag": self.dict_augmentation_flags[self.catalog[idx][3]]}

    def get(self, idx):
        np_file_idx = self.catalog[idx][0]
        if np_file_idx not in self._opened_np_file_idx_set:
            self._load_np_files({np_file_idx})
        return self._opened_np_file_data[np_file_idx][idx % self.entry_per_file]

    def get_batch(self, size=1000, pos=0):
        current = pos
        while True:
            start, end, count = self._calc_range(current, size)
            print start, end, count
            yield self._synth_ndarray_by_range(start, end)
            current += size
            if count < size:
                # 방금께 마지막이었음
                break


#
# class NpzArrayLoader(ISourceLoader):
#     pass

if __name__ == "__main__":
    NAL = NpyArrayLoader("g:\\test11.cat")
    print NAL.ranges
    a = 0
    for b in NAL.get_batch(size=9333, pos=0):
        print b.__len__()
        a += b.__len__()
    print a
