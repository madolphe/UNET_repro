from .decode_data import *

# @TODO: relative path would be better:
filename = '/Users/jouffroy/Desktop/thèse/UNET_repro/data/images.tfrecords'
batch_size = 5


def load_dataset(filename, batch_size):
    ds = dataset(filename)
    batched_dataset = ds.batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)
    return iterator
