# import tensorflow as tf
# from tensorflow.keras import Input
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda, MaxPool2D, ReLU, Reshape
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras import Model
# from tensorflow.keras.initializers import TruncatedNormal, Zeros
# import tensorflow.keras.backend as K
# import tensorflow_probability as tfp
# import numpy as np
# from operator import floordiv
# from tensorflow_addons.layers import InstanceNormalization
# import matplotlib.pyplot as plt
# # from utils import build_dataset
# import numpy as np
#
# input_size = (128, 128, 1)
# w_init = TruncatedNormal
# input_img = Input(shape=input_size, name='input_image')
# input_scope = Input(shape=input_size, name='input_scope')
# input_net = tf.concat((input_img, input_scope), axis=3)
#
# def down_block(prev_layer, filters, downsample=True):
#     conv = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=w_init(), use_bias=False)(prev_layer)
#     norm = InstanceNormalization()(conv)
#     acti = ReLU()(norm)
#
#     # down = MaxPool2D()(acti)
#     size = K.int_shape(acti)
#
#     down = tf.image.resize(acti, (size[1]//2, size[2]//2), method='nearest')
#
#     if downsample:
#         return down, acti
#     else:
#         return acti
#
#
# d1, s1 = down_block(input_net, 64)
# d2, s2 = down_block(d1, 128)
# d3, s3 = down_block(d2, 256)
# d4, s4 = down_block(d3, 512)
# s5 = down_block(d4, 1024, False)
# flat = Flatten()(s5)
# fully1 = Dense(128, activation='relu')(flat)
# fully2 = Dense(128, activation='relu')(fully1)
# fully3 = Dense(K.int_shape(flat)[1], activation='relu')(fully2)
# out = tf.reshape(fully3, K.shape(s5))
#
# def up_block(prev_layer, skip, filters, upsample=True):
#     cat = tf.concat((prev_layer, skip), axis=3)
#     conv = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=w_init(), use_bias=False)(cat)
#     norm = InstanceNormalization()(conv)
#     acti = ReLU()(norm)
#
#     size = K.int_shape(acti)
#
#     up = tf.image.resize(acti, (size[1]*2, size[2]*2), method='nearest')
#
#     if upsample:
#         return up
#     else:
#         return acti
#
# u1 = up_block(out, s5, 1024)
# u2 = up_block(u1, s4, 512)
# u3 = up_block(u2, s3, 256)
# u4 = up_block(u3, s2, 128)
# u5 = up_block(u4, s1, 64, False)
#
# output = Conv2D(2, (1, 1), strides=1, padding='same', kernel_initializer=w_init(), use_bias=False)(u5)
# log_output = tf.nn.sigmoid(output)
# #log_output = -tf.nn.softplus(-output)
# #log_output = tf.math.log(tf.math.sigmoid(output))
#
# model = Model([input_img, input_scope], output)
# model.summary()
# model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])
# model.fit(epochs=20, x=[x_train, np.ones((150, 128, 128, 1))], y=y_train)
