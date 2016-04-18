# -*- coding:utf-8 -*-

import re
import numpy as np


class PreProcess(object):
    def __init__(self, cmd, logger=None):
        self._cmd = cmd
        self._l = logger

        self._regex = re.compile("([\w\d\.\-]+)")
        self._chain = []

        self._parse()

    def log(self, msg):
        if not self._l:
            return
        self._l.i("[PreProcess] " + msg)

    def _parse(self):
        if self._cmd.__len__() == 0:
            return
        seps = self._cmd.split(",")
        for sep in seps:
            tokens = self._regex.findall(sep)

            try:
                func = "_" + tokens[0]
                f = self.__getattribute__(func)
                argc = f.func_code.co_argcount
                # 아규먼트가 1이면, 토큰은 2 이상이어야 한다.
                if tokens.__len__() < argc:
                    # 인자가 부족하다.
                    self.Log("Not enough arguments, Token : %s" % (sep,))
                    continue
                args = tokens[1:argc]
                desc = "func=%s, args=%s" % (func, str(args))
                self._chain.append((func, args, desc))
            except AttributeError:
                self.Log("No such function, Token : %s" % (sep,))
                continue

    def explain(self):
        if self._chain.__len__() == 0:
            return None
        desc_list = []
        for _, _, desc in self._chain:
            desc_list.append(desc)
        return " >> ".join(desc_list)

    def go(self, mat):
        if self._chain.__len__() == 0:
            return mat

        _mat = mat
        for func, args, _ in self._chain:
            _mat = self.__getattribute__(func)(_mat, *args)
        return _mat

    @staticmethod
    def _zcaw(mat, epsilon="0.1"):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape
        epsilon = float(epsilon)
        sigma = np.dot(mat, mat.T) / mat.shape[1]
        U, S, _ = np.linalg.svd(sigma)
        zca_mat = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)
        return np.dot(zca_mat, mat).reshape((shape[0], shape[1]))

    @staticmethod
    def _pcaw(mat):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape
        U, _, V = np.linalg.svd(mat)
        return np.dot(U, V).reshape((shape[0], shape[1]))

    @staticmethod
    def _gaussian_noise(mat, p="0.5"):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape

        p = float(p)
        n = 1
        s = np.random.binomial(n, p, shape[0] * shape[1])
        noised_mat = mat.reshape(shape[0] * shape[1])

        for i, out in enumerate(s):
            if out == 1:
                noised_mat[i] = 0

        return noised_mat.reshape((shape[0], shape[1]))

    @staticmethod
    def _histeq(mat):
        if not isinstance(mat, np.ndarray):
            return mat
        shape = mat.shape

        histogram = np.zeros(256)
        intensity = np.zeros(256)
        for r in range(shape[0]):
            for c in range(shape[1]):
                histogram[mat[r, c]] += 1.0
        for i in range(256):
            histogram[i] /= shape[0] * shape[1]
        intensity[0] = histogram[0]
        for i in range(1, 256):
            intensity[i] = intensity[i - 1] + histogram[i]
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        im = mat.reshape((shape[0] * shape[1]))
        for i in range(256):
            intensity[i] *= 255
        for i in range(shape[0] * shape[1]):
            img[i] = int(intensity[im[i]])

        return img.reshape((shape[0], shape[1]))
