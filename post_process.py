# -*- coding:utf-8 -*-
import os, sys
import argparse
import multiprocessing

from lib import CatParser, Log, PreProcess


class PostProcess(object):
    def __init__(self, input_cat, output_cat, update, worker_process_count, buffer_size, logger):
        try:
            self.c = CatParser(input_cat).export()
        except Exception, e:
            print str(e)
            raise

        self.source_path, _ = os.path.split(input_cat)
        self.np_files = self.c["np_files"]
        self.output_cat = output_cat
        self.update = update
        if worker_process_count > 0:
            # MKL 이 있는 경우 워커 프로세스가 1개인게 맞다.
            # 하지만 아닌경우 분할해서 처리하는것도 방법이다. 하지만 그건 고려하지 않겠다.
            self.worker_process_count = worker_process_count
        else:
            self.worker_process_count = multiprocessing.cpu_count()
        self.buffer_size = buffer_size

        self.l = logger

        self.worker_pool = []

    def _stand_by(self):
        pass

    def _clean_up(self):
        for p in self.worker_pool:
            p.join()
        self.l.i("[Np2Img] Cleaning Ok")

    def do(self):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="post_process")

    file_group = ap.add_argument_group("Essential")
    file_group.add_argument("-i", "--input_cat", type=str, default=None)
    file_group.add_argument("-o", "--output_cat", type=str, default=None)

    ap.add_argument("-u", "--update", action="store_true")

    ap.add_argument("-w", "--worker_process_count", type=int, default=1)
    # ap.add_argument("-t", "--io_thread_per_worker", type=int, default=2)
    # ap.add_argument("-f", "--output_image_format", type=str, default="PNG")
    # ap.add_argument("-c", "--output_image_colorspace", type=str, default="L")
    ap.add_argument("-b", "--buffer_size", type=int, default=65536)
    # ap.add_argument("-a", "--augmentation", type=str, default="")
    # ap.add_argument("-pp", "--postprocessing", type=str, default="")
    ap.add_argument("-l", "--logto", type=str, default="post_process.log")
    ap.add_argument("-L", "--loglevel", type=str, default="INFO")

    args = ap.parse_args()

    if not args.input_cat or not (args.output_cat or args.update):
        "--input_cat && (--output_cat || --update)"
        ap.print_help()
        sys.exit(0)

    with Log(args.logto, True, args.loglevel) as L:
        PP = PostProcess(args.input_cat, args.output_cat, args.update, args.worker_process_count,
                         args.buffer_size, L)
        PP.do()

        # np2img.py --input_cat g:\\t01.cat --output_path g:\\outimg
