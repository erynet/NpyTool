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

    def get_batch(self, size=1024, pos=0, shuffle=False, preprocess_cmd="", cachesize=0):
        return self._core.get_batch(size=size, pos=pos, shuffle=shuffle, \
                                    preprocess_cmd=preprocess_cmd, cachesize=cachesize)


class INpSource(object):
    def __init__(self, catalog_dict, source_path):
        self.cat = catalog_dict
        self.source_path = source_path

        self.total_count = self.cat["total_count"]
        self.entry_per_file = self.cat["entry_per_file"]
        self.parameter = self.cat["parameter"]
        self.dict_filenames = self.cat["dict_filenames"]
        self.dict_src_paths = self.cat["dict_src_paths"]
        self.np_files = self.cat["np_files"]
        self.ranges = self.cat["ranges"]
        self.catalog = self.cat["catalog"]

        self._opened_np_file_idx_set = set()
        self._opened_np_file_data = {}

        self._cache = {}
        self._cache_used_idx_Set = set()
        self._cachesize = 0
        self._index_list = None

        self._use_shuffle = False

    def _unload_np_files(self, idle_set=None, busy_set=None):
        raise NotImplementedError

    def _load_np_files(self, busy_set):
        raise NotImplementedError

    def _calc_range(self, pos, size):
        # 리턴값, is_linear, start, end, fpair, count

        epf = self.entry_per_file
        tc = self.total_count
        cat = self.catalog

        if self._use_shuffle:
            # 셔플 씀. -> 비선형
            flist = []
            il = self._index_list

            # 넘겨야 하는것 : 절대 인덱스, np_arr_idx, augmentation_idx, np_file_no
            for rel_index_list_idx, np_arr_idx in enumerate(il[pos:pos + size]):
                # pos + idx : 셔플된 index_list 의 절대 인덱스
                # d : 해당 index_list 의 내용물(np_array 에서의 인덱스)
                # np_file_dict[pos + rel_index_list_idx] = cat[np_arr_idx][0]
                flist.append((pos + rel_index_list_idx, np_arr_idx, 0, cat[np_arr_idx][0]))

            return False, None, None, flist, flist.__len__()
        else:
            # 셔플 안씀. -> 선형
            start = (cat[pos][0], pos % epf)
            if (pos + size - 1) < tc:
                end = (cat[pos + size - 1][0], (pos + size - 1) % epf)
                count = size
            else:
                end = (cat[tc - 1][0], (tc - 1) % epf)
                count = tc - pos
            return True, start, end, None, count

    def _synth_ndarray_by_linear(self, start, end):
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

    def _faluted_get(self, faluted, busy_set, cacheline=512):
        self._unload_np_files(busy_set=busy_set)
        self._load_np_files(busy_set=busy_set)
        result = {}
        epf = self.entry_per_file
        onfd = self._opened_np_file_data

        cache = self._cache
        cachesize = self._cachesize
        used = self._cache_used_idx_Set

        half_cacheline = cacheline / 2

        for idx, t in faluted:
            # abs_index_list_idx = t[0]
            np_arr_idx = t[1]
            rel_index_list_idx = np_arr_idx % epf
            # augmentation_flag_idx = t[2]
            np_file_no = t[3]

            result[idx] = onfd[np_file_no][rel_index_list_idx]

            cache_len = cache.__len__()
            if cache_len < cachesize:
                if cachesize - cache_len > cacheline:
                    # 7 + 나 + 8 =  16
                    bottom = rel_index_list_idx - half_cacheline
                    top = rel_index_list_idx + half_cacheline
                    if 0 > bottom:
                        bottom = 0
                    if epf < top:
                        top = epf

                    base_idx = (np_file_no * epf) + bottom
                    for iidx, d in enumerate(onfd[np_file_no][bottom:top]):
                        if (base_idx + iidx) in used:
                            continue
                        cache[base_idx + iidx] = d
        return result

    def _synth_ndarray_by_random(self, flist):
        # 1. 일단 flist 의 길이만큼 ndarray 를 만든다.
        # 2. 일단 flist 를 돌면서 캐쉬에 있나 없나부터 검색한다.
        # 3. 캐쉬에 없다고 확인된 넘들만 리스트에 넣고,
        # 4. 필요한 파일들을 일괄로 연다.

        busy_set = set()
        size = flist.__len__()
        result = {}
        cache = self._cache
        used = self._cache_used_idx_Set

        faluted = []
        for idx, t in enumerate(flist):
            abs_index_list_idx = t[0]
            v = cache.pop(abs_index_list_idx, None)
            if v is not None:
                result[idx] = v
                used.add(idx)
            else:
                faluted.append((idx, t))
                busy_set.add(t[3])
        result.update(self._faluted_get(faluted=faluted, busy_set=busy_set))
        return np.concatenate([[result[i] for i in xrange(size)]])

    def get_metadata(self, idx):
        return {"src_path": self.dict_src_paths[self.catalog[idx][1]],
                "src_filename": self.dict_filenames[self.catalog[idx][2]],
                "np_file": self.np_files[self.catalog[idx][0]],
                "parameter": self.parameter}
                # "augmentation_flag": self.dict_augmentation_flags[self.catalog[idx][3]]}

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
        return self._synth_ndarray_by_linear(start, end)

    def _reset_batch_env(self):
        # 배치관련 변수 초기화
        self._opened_np_file_idx_set = set()
        self._opened_np_file_data = {}

        self._cache = {}
        # self._cache_used_idx_Set = set()
        self._cachesize = 0
        self._index_list = None
        gc.collect()

        self._aug_flags = ("O",)
        self._aug_power = 1

        self._use_augmentation = False
        self._use_shuffle = False

    def get_batch(self, size=1024, pos=0, shuffle=False, preprocess_cmd="", cachesize=0):
        self._reset_batch_env()

        self._index_list = range(self.total_count)
        if shuffle:
            import random
            random.shuffle(self._index_list)
            self._use_shuffle = True
        il = self._index_list

        if cachesize > 0:
            self._cachesize = cachesize
        else:
            # 기본값은 전체 셋의 75% 를 할당하되 ... 12.5만개를 리미트로 삽는다.
            if int(il.__len__() * 0.75) <= 128000:
                self._cachesize = int(il.__len__() * 0.75)
            else:
                self._cachesize = 128000

        current = pos
        batch_round = 0
        while True:
            is_linear, start, end, flist, count = self._calc_range(current, size)

            if is_linear:
                yield batch_round, self._synth_ndarray_by_linear(start, end)
            else:
                yield batch_round, self._synth_ndarray_by_random(flist)
            batch_round += 1
            gc.collect()
            current += size
            if count < size:
                break
        self._reset_batch_env()


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
    NAL = NpArrayLoader(u"G:\\output\\t18.cat")

    # for i in range(30):
    #     print i
    a = 0
    for r, b in NAL.get_batch(size=333, pos=0, shuffle=True):
        # a += b.__len__()
        print "round : %d, size : %d" % (r, b.__len__())
        # print b.shape
        # break
        # print "round : %d" % (r,)
        continue
        # print "total : %d" % (a,)
    print "elapsed ms : %.2f" % ((time.time() - _start) * 1000,)

    # time.sleep(3)
