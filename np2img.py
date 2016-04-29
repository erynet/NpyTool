# -*- coding:utf-8 -*-

import sys, os
import multiprocessing

import imp

imp.reload(sys)
sys.setdefaultencoding("utf-8")

import time
import argparse

try:
    import ujson as json
except ImportError:
    import pip

    pip.main(["install", "ujson==1.35"])
    import ujson as json

from lib import CatParser, Log


class Np2Img(object):
    def __init__(self, input_cat, output_path, worker_process_count, io_thread_per_worker, \
                 output_image_format, output_image_colorspace, buffer_size, logger):
        try:
            self.c = CatParser(input_cat).export()
        except Exception, e:
            print str(e)
            raise

        self.source_path, _ = os.path.split(input_cat)
        self.np_files = self.c["np_files"]
        self.output_path = output_path
        if worker_process_count > 0:
            self.worker_process_count = worker_process_count
        else:
            self.worker_process_count = multiprocessing.cpu_count()
        self.io_thread_per_worker = io_thread_per_worker
        self.output_image_format = output_image_format
        self.output_image_colorspace = output_image_colorspace
        self.buffer_size = buffer_size

        self.l = logger

        self.worker_pool = []
        # self.consumer = None

    def _stand_by(self):
        # 미리 타겟 폴더들을 다 만든다.
        self.l.i("[Np2Img] Stand by ... !!")
        self.target_paths = []
        for rel_path in self.c["dict_src_paths"]:

            abs_path = os.path.join(self.output_path, rel_path)
            # print self.out_path, rel_path, abs_path
            self.target_paths.append(abs_path)
            try:
                if os.path.exists(abs_path):
                    continue
                os.makedirs(abs_path, mode=755)
                self.l.i("[Np2Img] Directory Created, %s" % (abs_path,))
            except Exception, e:
                self.l.i("[Np2Img] Directory Creation failed, reason : %s" % (e.message,))
                continue

    def _clean_up(self):
        # self.consumer.join()
        for p in self.worker_pool:
            p.join()
        self.l.i("[Np2Img] Cleaning Ok")

    def export(self):
        def make_chunk(l, n):
            k, m = len(l) / n, len(l) % n
            return (l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

        from lib import Np2ImgWorker
        self._stand_by()

        luggage = []
        for i in range(self.c["catalog"].__len__()):
            luggage.append((i, self.c["catalog"][i]))

        worker_kwargs = {"source_path": self.source_path,
                         "np_files": self.np_files,
                         "base_path": self.output_path,
                         "rel_paths": self.target_paths,
                         "filenames": self.c["dict_filenames"],
                         "augmentaion_flags": self.c["dict_augmentation_flags"],
                         "io_thread_count": self.io_thread_per_worker,
                         "logger": self.l,
                         # "compressed": self.c["compress"],
                         "image_format": self.output_image_format,
                         "colorspace": self.output_image_colorspace,
                         "luggage": None,
                         "entry_per_file": self.c["entry_per_file"],
                         "instance_index": 0}
        idx = 0
        for l in (l for l in make_chunk(luggage, self.worker_process_count)):
            worker_kwargs["luggage"] = l
            worker_kwargs["instance_index"] = idx
            self.worker_pool.append(multiprocessing.Process(target=Np2ImgWorker, kwargs=worker_kwargs))
            idx += 1

        for p in self.worker_pool:
            p.start()

        self._clean_up()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="img2np")

    file_group = ap.add_argument_group("Essential")
    file_group.add_argument("-i", "--input_cat", type=str, default=None)
    file_group.add_argument("-o", "--output_path", type=str, default=None)

    ap.add_argument("-w", "--worker_process_count", type=int, default=0)
    ap.add_argument("-t", "--io_thread_per_worker", type=int, default=2)
    ap.add_argument("-f", "--output_image_format", type=str, default="PNG")
    ap.add_argument("-c", "--output_image_colorspace", type=str, default="L")
    ap.add_argument("-b", "--buffer_size", type=int, default=65536)
    # ap.add_argument("-a", "--augmentation", type=str, default="")
    # ap.add_argument("-pp", "--postprocessing", type=str, default="")
    ap.add_argument("-l", "--logto", type=str, default="np2img.log")
    ap.add_argument("-L", "--loglevel", type=str, default="INFO")

    args = ap.parse_args()

    if not args.input_cat or not args.output_path:
        ap.print_help()
        sys.exit(0)

    if args.output_image_colorspace not in ("L", "RGB", "CMYK"):
        print "colorspace is must one of `L, RGB, CMYK`"
        sys.exit(0)

    try:
        with Log(args.logto, True, args.loglevel) as L:
            N2I = Np2Img(args.input_cat, args.output_path, args.worker_process_count, args.io_thread_per_worker, \
                         args.output_image_format, args.output_image_colorspace, args.buffer_size, L)
            N2I.export()
    except KeyboardInterrupt as e:
        # killing Signal
        pass

    # np2img.py --input_cat g:\\t01.cat --output_path g:\\outimg
