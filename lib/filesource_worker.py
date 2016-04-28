# -*- coding:utf-8 -*-

import os, sys
import time
import bz2
import Queue

try:
    import numpy as np
except ImportError:
    import pip

    pip.main(["install", "numpy"])
    import numpy as np

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

from lib import Augmentation, PreProcess

# io 스레드와 워커를 붅리할것. (?)


class FileSourceWorker(object):
    def __init__(self, root_path, file_name_with_path_but_ext, length_of_side, dtype, total_size, max_entry_count, \
                 augmentation_cmd, pre_process_cmd, logger, in_q, out_q, instance_index, compress, role):
        self._root_path = root_path
        self._file_name_with_path_but_ext = file_name_with_path_but_ext
        self._length_of_side = length_of_side
        self._dtype = dtype

        self._total_size = total_size
        self._max_entry_count = max_entry_count

        self._augmentation_cmd = augmentation_cmd
        self._augmentation = Augmentation(cmd=self._augmentation_cmd, logger=logger)
        self._pre_process_cmd = pre_process_cmd
        self._pre_process = PreProcess(cmd=self._pre_process_cmd, logger=logger)

        self._l = logger
        self._in_q = in_q
        self._out_q = out_q

        self._instance_index = instance_index

        if role == 0:
            self._producer_instance()
        elif role == 1:
            self._consumer_instance(compress, 3, 2)

    def _producer_instance(self):
        idx = self._instance_index
        side = self._length_of_side

        count = 0
        exported = 0

        au = self._augmentation
        au_desc, power = au.explain()
        pp = self._pre_process
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
                # print mats.__len__()
                for aug in mats:
                    # 전처리 _chain 구동
                    results.append((aug, pp.go(mats[aug])))
            except IOError, e:
                self._l.e("[Producer #%2d] IOError, %s, %s" % (idx, full_path, e.message))
                continue
            except Exception, e:
                self._l.e("[Producer #%2d] Exception, %s, %s" % (idx, full_path, e.message))
                continue
            finally:
                if "img" in __name__ and img:
                    img.close()

            dt = float((time.time() - _start) * 1000. / power)
            self._l.d("[Producer #%2d] %s, processed" % (idx, full_path))
            rel_path = os.path.relpath(path, self._root_path)
            for result in results:
                exported += 1
                self._out_q.put((rel_path, fn, result[0], dt, result[1]))
            count += 1
        self._l.i("[Producer #%2d] Exit, Processed Count %d, Exported Count %d" % (idx, count, exported))

    def _consumer_instance(self, compress=False, wait_sec_for_producer=5, timeout_sec=3):
        def _progress():
            mean = float(np.mean(_st))
            progress = (total_count * 100) / total_entry
            dt = time.time() - _start
            estimation = (dt / total_count) * total_entry
            # print float(np.mean(_st)), _st
            self._l.i("[Consumer] [%3d%%] %.5f ms per image, %.2f/%.2f sec" % (progress, mean, dt, estimation))

        def _dump():
            if compress:
                if max_entry <= total_entry:
                    np_fn = "%s_%05d.npz" % (filename_base, rotate)
                else:
                    np_fn = "%s.npz" % (filename_base,)
                self._l.i("[Consumer] Compress start ...")
                _compress_start = time.time()
                np.savez_compressed(np_fn, arr=_arr)
                self._l.i("[Consumer] Compress end (%.2fs) -> %s" % (time.time() - _compress_start, np_fn))
            else:
                if max_entry <= total_entry:
                    np_fn = filename_base + "_%05d" % rotate + ".npy"
                else:
                    np_fn = filename_base + ".npy"
                _dump_start = time.time()
                np.save(np_fn, _arr)
                self._l.i("[Consumer] Dump done (%.2fs) -> %s" % (time.time() - _dump_start, np_fn))

            _, filename = os.path.split(np_fn)
            _cat["np_files"].append(filename)
            _cat["ranges"].append((total_count - (total_count % max_entry), total_count))

        def _dump_catalog():
            _cat["total_count"] = total_count

            cat_fn = filename_base + ".cat"
            with open(cat_fn, "wb") as fp:
                # fp.write(bz2.compress(json.dumps(_cat), 9))
                fp.write(json.dumps(_cat, ensure_ascii=False))
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

        _cat = {"parameter": {"augmentation:": self._augmentation.cmd(), "preprocessing": self._pre_process.cmd()}, \
                "np_files": [], "ranges": [], \
                "dict_augmentation_flags": [], "dict_src_paths": [], "dict_filenames": [], \
                "catalog": [], \
                "total_count": 0, \
                "entry_per_file": max_entry, \
                "compress": compress}
        _st = []

        _aug_index = {}
        _path_index = {}
        _filename_index = {}

        if max_entry > total_entry:
            _arr = np.zeros(shape=(total_entry, length_of_side, length_of_side), dtype=self._dtype)
        else:
            _arr = np.zeros(shape=(max_entry, length_of_side, length_of_side), dtype=self._dtype)

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

            if path not in _path_index:
                _cat["dict_src_paths"].append(path)
                _path_index[path] = (_cat["dict_src_paths"].__len__() - 1)

            if fn not in _filename_index:
                _cat["dict_filenames"].append(fn)
                _filename_index[fn] = (_cat["dict_filenames"].__len__() - 1)

            if aug not in _aug_index:
                _cat["dict_augmentation_flags"].append(aug)
                _aug_index[aug] = (_cat["dict_augmentation_flags"].__len__() - 1)

            _cat["catalog"].append((rotate, _path_index[path], _filename_index[fn], _aug_index[aug]))

            _st.append(dt)
            _arr[count] = mat

            # if total_count > 0 and (total_count % ((total_entry / 100) * 3)) == 0:
            if total_count > 0 and ((total_count * 100) % (total_entry * 3)) / 100 == 0:
                _progress()

            if count > 0 and ((count % max_entry) == (max_entry - 1)):
                _dump()

                count = 0
                rotate += 1

                if total_entry - total_count >= max_entry:
                    _arr = np.zeros(shape=(max_entry, length_of_side, length_of_side), dtype=self._dtype)
                else:
                    _arr = np.zeros(shape=(total_entry - total_count, length_of_side, length_of_side), dtype=self._dtype)
            else:
                count += 1

            total_count += 1

        _dump_catalog()
        self._l.i("[Consumer] total %d elements processed" % (total_count,))
        self._l.i("[Consumer] Exit")
