import tensorflow as tf
import PIL
import numpy as np
import os

mylist = os.listdir("./data")


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized


dataset = tf.data.Dataset.from_tensor_slices((mylist))
dataset = dataset.map(_parse_function)
Bdataset = dataset.batch(10)

iterator = Bdataset.make_one_shot_iterator()
next_element = iterator.get_next()


print(dataset)
print(Bdataset)
print(iterator)
print(next_element)
