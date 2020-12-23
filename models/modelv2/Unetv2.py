import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Conv2D, Flatten, Dense, ReLU, Input
from tensorflow_addons.layers import InstanceNormalization
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import numpy as np


class Unet:
    def __init__(self, input_size):
        self.input_img = Input(shape=input_size)
        # self.input_scope = Input(shape=input_size, name='input_scope')
        # self.input_net = tf.concat((self.input_img, self.input_scope), axis=3)
        self.w_init = TruncatedNormal
        # self.model = Model([self.input_img, self.input_scope], self.create_unet)
        self.model = Model(self.input_img, self.create_unet())
        # self.model.summary()

    def down_block(self, prev_layer, filters, downsample=True):
        conv = Conv2D(filters, (3, 3), strides=1, padding='same',
                      kernel_initializer=self.w_init(), use_bias=False)(prev_layer)
        norm = InstanceNormalization()(conv)
        acti = ReLU()(norm)
        size = K.int_shape(acti)
        down = tf.image.resize(acti, (size[1]//2, size[2]//2), method='nearest')
        if downsample:
            return down, acti
        else:
            return acti

    def up_block(self, prev_layer, skip, filters, upsample=True):
        cat = tf.concat((prev_layer, skip), axis=3)
        conv = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=self.w_init(), use_bias=False)(cat)
        norm = InstanceNormalization()(conv)
        acti = ReLU()(norm)
        size = K.int_shape(acti)
        up = tf.image.resize(acti, (size[1]*2, size[2]*2), method='nearest')
        if upsample:
            return up
        else:
            return acti

#    @property
    def create_unet(self):
        d1, s1 = self.down_block(self.input_img, 64)
        d2, s2 = self.down_block(d1, 128)
        d3, s3 = self.down_block(d2, 256)
        d4, s4 = self.down_block(d3, 512)
        s5 = self.down_block(d4, 1024, False)
        flat = Flatten()(s5)
        fully1 = Dense(240, activation='relu')(flat)
        fully2 = Dense(240, activation='relu')(fully1)
        fully3 = Dense(K.int_shape(flat)[1], activation='relu')(fully2)
        out = tf.reshape(fully3, K.shape(s5))
        u1 = self.up_block(out, s5, 1024)
        u2 = self.up_block(u1, s4, 512)
        u3 = self.up_block(u2, s3, 256)
        u4 = self.up_block(u3, s2, 128)
        u5 = self.up_block(u4, s1, 64, False)
        output = Conv2D(2, (1, 1), strides=1, padding='same', kernel_initializer=self.w_init(), use_bias=False)(u5)
        return output
