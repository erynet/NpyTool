# -*- coding:utf-8 -*-

from .log import *
from .augmentation import *
from .preprocess import *
from .filesource_worker import *
from .np2img_worker import *


class CatParser(object):
    def __init__(self, catalog):
        if not os.path.exists(catalog):
            raise IOError("There are no such file")
        try:
            with open(catalog, "rb") as fp:
                self.c = json.loads(fp.read())
        except IOError, e:
            print catalog
            print str(e)
            raise IOError(e.message)
        except Exception, e:
            print str(e)
            raise

    def export(self):
        return self.c
