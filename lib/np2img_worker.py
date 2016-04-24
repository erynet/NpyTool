# -*- coding:utf-8 -*-

import os
import time
import threading, Queue
import numpy as np
try:
    from PIL import Image
except ImportError:
    import pip

    pip.main(["install", "Pillow==3.2.0"])
    from PIL import Image
from StringIO import StringIO

import gc


class Np2ImgWorker(object):
    def __init__(self, source_path, np_files, base_path, rel_paths, filenames, augmentaion_flags, \
                 io_thread_count, logger, image_format, colorspace, luggage, entry_per_file, instance_index):
        self.source_path = source_path
        self.np_files = np_files
        self.base_path = base_path
        self.rel_paths = rel_paths
        self.filenames = filenames
        self.augmentaion_flags = augmentaion_flags
        self.l = logger
        self.image_format = image_format.lower()
        self.image_extension = ""
        self.colorspace = colorspace
        self.luggage = luggage
        self.entry_per_file = entry_per_file
        self.instance_index = instance_index

        self.thread_pool = []
        self.shared_dict = {}

        self._worker_instance(instance_index, io_thread_count)

    def _io_thread(self, pid, idx, queue):
        pidx = pid
        tidx = idx

        def i(msg):
            self.l.i("[Np2ImgWorker IOThread %5d:%2d] %s" % (pidx, tidx, msg))

        def d(msg):
            self.l.d("[Np2ImgWorker IOThread %5d:%2d] %s" % (pidx, tidx, msg))

        def w(msg):
            self.l.w("[Np2ImgWorker IOThread %5d:%2d] %s" % (pidx, tidx, msg))
        i("Init")

        start = time.time()
        count = 0
        while True:
            try:
                idx = queue.get(block=True, timeout=0.1)
            except Queue.Empty:
                continue
            if idx is None:
                i("Received Terminate Signal")
                break
            target = self.shared_dict[idx]

            fn, _ = os.path.splitext(target[1])
            if target[2][1:].__len__() == 0:
                fn = "%s.%s" % (fn, self.image_extension)
            else:
                fn = "%s.%s.%s" % (fn, target[2][1:], self.image_extension)

            full_path = os.path.join(target[0], fn)
            try:
                with open(os.path.join(full_path), "wb") as fp:
                    fp.write(target[3])
                    fp.flush()
            except IOError, e:
                print full_path
                w("#1 IOError : %s" % (str(e)))
            except Exception, e:
                w("#2 Exception : %s" % (str(e)))
            target = None
            del self.shared_dict[idx]
            d("%s written" % (full_path,))
            count += 1
        i("Exit, %d files processed" % (count,))

    def _worker_instance(self, instance_index, io_thread_count):
        def i(msg):
            self.l.i("[Np2ImgWorker %2d] %s" % (instance_index, msg))

        def d(msg):
            self.l.d("[Np2ImgWorker %2d] %s" % (instance_index, msg))

        def w(msg):
            self.l.w("[Np2ImgWorker %2d] %s" % (instance_index, msg))

        save_kwargs = {}
        if self.image_format == "bmp":
            save_kwargs["format"] = "BMP"
            self.image_extension = "bmp"
        elif self.image_format == "png":
            save_kwargs["format"] = "PNG"
            self.image_extension = "png"
        elif self.image_format == "jpg" or self.image_format == "jpeg":
            save_kwargs["format"] = "JPEG"
            save_kwargs["quality"] = 90
            self.image_extension = "jpg"
        else:
            save_kwargs["format"] = "PNG"
            self.image_extension = "png"
        i("Image Export Format is `%s`" % (save_kwargs["format"],))

        opened_np_file_set = set()
        opened_np_data = {}

        def load_np(busy):
            if busy in opened_np_file_set:
                return
            opened_np_data[busy] = np.load(os.path.join(self.source_path, self.np_files[busy]), "r")
            d("%s loaded" % (self.np_files[busy],))
            opened_np_file_set.update({busy})

        def unload_np(busy):
            for idx in opened_np_file_set.difference({busy}):
                d("%s unloaded" % (self.np_files[idx],))
                del opened_np_data[idx]
            gc.collect()
            opened_np_file_set.intersection_update({busy})

        i("Init (pid: %5d)" % (os.getpid(),))

        self.luggage.sort(key=lambda lu: lu[1][0])

        q = Queue.Queue(-1)

        for idx in range(io_thread_count):
            self.thread_pool.append(threading.Thread(target=self._io_thread, args=(os.getpid(), idx, q)))
        for t in self.thread_pool:
            t.start()

        epf = self.entry_per_file

        for idx, l in self.luggage:
            try:
                rot = l[0]
                path = os.path.join(self.base_path, self.rel_paths[l[1]])
                filename = self.filenames[l[2]]
                aug = self.augmentaion_flags[l[3]]
                unload_np(rot)
                load_np(rot)

                im = Image.fromarray(opened_np_data[rot][idx % epf])
                fake_fp = StringIO()
                im.convert(self.colorspace).save(fp=fake_fp, **save_kwargs)
                self.shared_dict[idx] = (path, filename, aug, fake_fp.getvalue())

                q.put(idx)
            except IOError, e:
                w("#3 IOError : %s" % (str(e)))
            except Exception, e:
                w("#4 Exception : %s" % (str(e)))

        i("Killing Threads begin ...")
        for _ in range(io_thread_count):
            q.put(None)
        for t in self.thread_pool:
            t.join()
        i("Killing Threads done")
        i("Exit")

# class ValidationLogger:
#     pass
#
# class NpArrayLoader:
#     pass

# with ValidationLogger(name="Psudo set with option .... @#!@#", params="histeq, ... ... some option") as VL:
#     # do some learning
#     VL.SetValidationMethod("A 0.1,B 7.2,C ...")
#
#     VALSET = NpArrayLoader(u"G:\\output\\t02.cat")
#
#     for b in VALSET.get_batch(size=10, pos=0, augmentation=False, shufle=False):
#         for bs in b:
#
#             # do some reconstruct
#
#             VL.Report(bs, recon_bs)


# cmd = ['F', 'FR090', 'FR180', 'FR270', 'FT', 'FTR090', 'FTR180', 'FTR270', 'R090', 'R180', 'R270', 'T', 'TR090', 'TR180', 'TR270']
#
#
# def aug(cmd, mat):
#     c = cmd
#     m = mat
#     for _ in range(5):
#         if c[0] == "F":
#             c = c[1:]
#             m = np.fliplr(m)
#         elif c[0] == "T":
#             c = c[1:]
#             m = np.transpose(m)
#         elif c[0] == "R":
#             if c[1:] == "090":
#                 c = c[4:]
#                 m = np.rot90(m, 1)
#             elif c[1:] == "180":
#                 c = c[4:]
#                 m = np.rot90(m, 2)
#             elif c[1:] == "270":
#                 c = c[4:]
#                 m = np.rot90(m, 3)
#     return m
