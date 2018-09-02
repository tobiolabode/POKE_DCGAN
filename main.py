import tensorflow as tf
import PIL
import numpy as np
import os


def preprocess():
    # os.mkdir(dst)
    for each in os.listdir(src):
        img = PIL.imread(os.path.join(src, each))
        img = PIL.resize(img, (256, 256))
        PIL.imwrite(os.path.join(dst, each), img)
