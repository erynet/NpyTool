# -*- coding:utf-8 -*-

import re
import numpy as np


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
