import tensorflow as tf
import PIL
import numpy as np
import os


def preprocess():
    # os.mkdir(dst)
    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized
