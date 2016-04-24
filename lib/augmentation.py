# -*- coding:utf-8 -*-

import re
import numpy as np


class Augmentation(object):
    def __init__(self, cmd, logger=None):
        self._cmd = cmd.lower()
        self._l = logger

        self._regex = re.compile("([ftr]{1})")
        self._chain = []

        self._flags = []
        self._power = 1
        self.parse(cmd)

    def log(self, msg):
        if not self._l:
            return
        self._l.i("[Augmentation] " + msg)

    def parse(self, cmd):
        if cmd.__len__() == 0:
            self._flags = ["O"]
            return
        tokens = self._regex.findall(self._cmd)
        if tokens.__len__() == 0:
            self._flags = ["O"]
            return
        flags = ["O"]

        if "f" in tokens:
            new_flags = []
            for flag in flags:
                new_flags.append(flag + "F")
            flags += new_flags
            # flip
            self._chain.append(("_flip", "Flip", 2, "f"))
        if "t" in tokens:
            new_flags = []
            for flag in flags:
                new_flags.append(flag + "T")
            flags += new_flags
            # transpose
            self._chain.append(("_transpose", "Transpose", 2, "t"))
        if "r" in tokens:
            new_flags = []
            for flag in flags:
                new_flags.append(flag + "R090")
                new_flags.append(flag + "R180")
                new_flags.append(flag + "R270")
            flags += new_flags
            # rotation
            self._chain.append(("_rotate", "Rotate", 4, "r"))
        if flags.__len__() == 1:
            raise ValueError("Invalid augmentation command : %s" % (self._cmd,))

        for idx, flag in enumerate(flags):
            if flag == "OFTR270":
                flags[idx] = "O"
            elif flag == "OTR270":
                flags[idx] = "OF"
            elif flag == "OFR090":
                flags[idx] = "OT"
            elif flag == "OFR180":
                flags[idx] = "OTR090"
            elif flag == "OFR270":
                flags[idx] = "OTR180"
            elif flag == "OFT":
                flags[idx] = "OR090"
            elif flag == "OFTR090":
                flags[idx] = "OR180"
            elif flag == "OFTR180":
                flags[idx] = "OR270"
        flags = list(set(flags))
        flags.sort()
        self._power = flags.__len__()
        self._flags = tuple(flags)

    def cmd(self):
        return "".join(map(lambda x: x[3], self._chain))

    def flags(self):
        return self._flags

    @staticmethod
    def calc_power(cmd):
        if cmd.__len__() == 0:
            return 1
        tokens = re.findall("([ftr]{1})", cmd)
        if tokens.__len__() == 0:
            return 1
        flags = ["O"]

        if "f" in tokens:
            new_flags = []
            for flag in flags:
                new_flags.append(flag + "F")
            flags += new_flags
        if "t" in tokens:
            new_flags = []
            for flag in flags:
                new_flags.append(flag + "T")
            flags += new_flags
        if "r" in tokens:
            new_flags = []
            for flag in flags:
                new_flags.append(flag + "R090")
                new_flags.append(flag + "R180")
                new_flags.append(flag + "R270")
            flags += new_flags

        for idx, flag in enumerate(flags):
            if flag == "OFTR270":
                flags[idx] = "O"
            elif flag == "OTR270":
                flags[idx] = "OF"
            elif flag == "OFR090":
                flags[idx] = "OT"
            elif flag == "OFR180":
                flags[idx] = "OTR090"
            elif flag == "OFR270":
                flags[idx] = "OTR180"
            elif flag == "OFT":
                flags[idx] = "OR090"
            elif flag == "OFTR090":
                flags[idx] = "OR180"
            elif flag == "OFTR180":
                flags[idx] = "OR270"
        return set(flags).__len__()

    def explain(self):
        if self._chain.__len__() == 0:
            return None, 1
        desc_list = []
        p = self._power
        for _, desc, power, _ in self._chain:
            desc_list.append(desc)
        return " >> ".join(desc_list) + " (x%d)" % p, p

    def go_batch(self, in_mats):
        if self._flags.__len__() == 0:
            if isinstance(in_mats, (tuple, list)):
                return in_mats
            else:
                return [in_mats]
        import copy

        if not isinstance(in_mats, (tuple, list)):
            in_mats = [in_mats]

        out_mats = []
        for mat in in_mats:
            for c in self._flags:
                m = copy.deepcopy(mat)
                for _ in range(5):
                    if not c.__len__():
                        break

                    if c[0] == "O":
                        c = c[1:]
                    elif c[0] == "F":
                        c = c[1:]
                        m = np.fliplr(m)
                    elif c[0] == "T":
                        c = c[1:]
                        m = np.transpose(m)
                    elif c[0] == "R":
                        c = c[1:]
                        if c == "090":
                            c = c[3:]
                            m = np.rot90(m, 1)
                        elif c == "180":
                            c = c[3:]
                            m = np.rot90(m, 2)
                        elif c == "270":
                            c = c[3:]
                            m = np.rot90(m, 3)
                out_mats.append(m)
        return out_mats

    def go(self, mat):
        if self._flags.__len__() == 0:
            return {"O": mat}
        import copy
        _mats = {}
        for f in self._flags:
            # print c
            m = copy.deepcopy(mat)
            c = copy.copy(f)
            for _ in range(5):
                if not c.__len__():
                    break

                if c[0] == "O":
                    c = c[1:]
                elif c[0] == "F":
                    c = c[1:]
                    m = np.fliplr(m)
                elif c[0] == "T":
                    c = c[1:]
                    m = np.transpose(m)
                elif c[0] == "R":
                    c = c[1:]
                    if c == "090":
                        c = c[3:]
                        m = np.rot90(m, 1)
                    elif c == "180":
                        c = c[3:]
                        m = np.rot90(m, 2)
                    elif c == "270":
                        c = c[3:]
                        m = np.rot90(m, 3)
            _mats[f] = m
        # for func, _, _, _ in self._chain:
        #     _mats.update(self.__getattribute__(func)(_mats))
        # print _mats.keys()
        return _mats

    # @staticmethod
    # def _flip(mats):
    #     new_mats = {}
    #     for k in mats:
    #         new_mats[k + "F"] = np.fliplr(mats[k])
    #     return new_mats
    #
    # @staticmethod
    # def _transpose(mats):
    #     new_mats = {}
    #     for k in mats:
    #         new_mats[k + "T"] = np.transpose(mats[k])
    #     return new_mats
    #
    # @staticmethod
    # def _rotate(mats):
    #     new_mats = {}
    #     for k in mats:
    #         for c in range(1, 4):
    #             new_mats[k + "R%03d" % (90 * c)] = np.rot90(mats[k], c)
    #     return new_mats
