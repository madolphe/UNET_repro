import tensorflow as tf
import tensorflow_datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import sys
from .decode_data import *
from tensorflow.compat.v1.data import make_one_shot_iterator

filename = '/Users/jouffroy/Desktop/thèse/UNET_repro/data/images.tfrecords'
batch_size = 5



def load_dataset(filename, batch_size):
    ds = dataset(filename)
    batched_dataset = ds.batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)
    return iterator


if __name__ == "__main__":
    iterator = load_dataset(filename, 5)
    mask = next(iterator)["mask"]
    plt.imshow(mask[0,0,:,:,0])
    plt.show()
    plt.imshow(mask[0,1,:,:,0])
    plt.show()
    plt.imshow(mask[0,2,:,:,0])
    plt.show()