# -*- coding:utf-8 -*-

import sys, os
import threading

import imp

imp.reload(sys)
sys.setdefaultencoding("utf-8")

import time
from datetime import datetime, timedelta
import re, zlib, bz2, base64
from struct import pack, unpack
import copy

try:
    import numpy as np
except ImportError:
    import pip
    pip.main(["install", "numpy==1.11.0"])
    import numpy as np

try:
    import mysql.connector as mc
except ImportError:
    import pip
    pip.main(["install", "mysql-connector-python-rf==2.1.3"])
    import mysql.connector as mc

# import logging
# import logging.handlers

import argparse


class NpyTool(object):
    def __init__(self):
        pass


# 1. 특정 폴더의 파일들 -> .npy, .cat
# 2. 특정 폴더의 파일들 -> DB
# 3. DB 의 특정 파일 혹은 그룹 -> .npy

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="NpyTool")
    op_group = ap.add_mutually_exclusive_group(required=True)
    op_group.add_argument("-i", "--import", action="store_true")
    op_group.add_argument("-e", "--export", action="store_true")
    #
    # me_group = ap.add_mutually_exclusive_group(required=True)

    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("-p", "--frompath", action="store_true")
    source_group.add_argument("-d", "--fromdb", action="store_true")

    file_group = ap.add_argument_group("From File")
    # file_group.add_argument("-p", "--frompath", action="store_true")
    file_group.add_argument("-P", "--path", type=str, default=None)
    file_group.add_argument("-c", "--concurrent_input", type=int, default=1)
    file_group.add_argument("-r", "--regex_pattern_of_file", type=str, default="*+\.*+")
    #
    db_group = ap.add_argument_group("From Database")
    # db_group.add_argument("-d", "--fromdb", action="store_true")
    # # db_group.add_argument("--mysqlsock", type=str, default="/var/run/mysqld/mysqld.sock")
    db_group.add_argument("--mysqlhost", type=str, default="evancho.ery.wo.tc")
    db_group.add_argument("--mysqlport", type=int, default=33007)
    db_group.add_argument("--mysqluser", type=str, default="leatherdb_reader")
    db_group.add_argument("--mysqlpasswd", type=str, default="a8d48e84e1e4424272e39962330f4eacbb7302073b6d9562ede4127ed8a39b84")

    ap.add_argument("-l", "--logto", type=str, default="NpyTool.log")
    ap.add_argument("-L", "--loglevel", type=str, default="INFO")

    args = ap.parse_args()




    # op_group = ap.add_argument_group("Operation")
    # db_group = ap.add_argument_group("Database")

    # ap.add_argument("-m", "--port", type=str, default="")
    # ap.add_argument("-c", "--concurrent_input", type=int, default=1)
    # ap.add_argument("-o", "--out", type=str, default="NpyTool.log")
    # ap.add_argument("-l", "--loglevel", type=str, default="INFO")
    # ap.add_argument("--mysqlsock", type=str, default="/var/run/mysqld/mysqld.sock")
    # ap.add_argument("--mysqlhost", type=str, default="evancho.ery.wo.tc")
    # ap.add_argument("--mysqlport", type=int, default=33006)
    # ap.add_argument("--mysqluser", type=str, default="coock_local_acd")
    # ap.add_argument("--mysqlpasswd", type=str, default="")