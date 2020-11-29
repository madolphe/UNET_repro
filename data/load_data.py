import tensorflow as tf
import tensorflow_datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import sys
import decode_data
from tensorflow.compat.v1.data import make_one_shot_iterator

filename = '/Users/jouffroy/Desktop/thèse/UNET_repro/data/images.tfrecords'
batch_size = 5

dataset = decode_data.dataset(filename)
batched_dataset = dataset.batch(batch_size)
iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)
data = iterator.get_next()
img = np.array(data['image'][0])
mask = np.array(data['mask'][0, 0, :, :])
plt.imshow(img)
plt.show()
plt.imshow(mask)
plt.show()


if __name__ == '__main__':
    pass
