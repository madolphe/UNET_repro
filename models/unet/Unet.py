import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from operator import floordiv 


class Unet(tf.keras.Model):
    def __init__(self, output_classes, img_size, batch_size, input_shape):
        super().__init__()
        self.output_classes = output_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.create_unet(input_shape)

    def __call__(self, x, training=False):
        
        x = self.conv1(x)
        x_2 = self.conv2(x)
        x = self.pool_layer(x_2)
        #
        x = self.conv3(x)
        x_4 = self.conv4(x)
        x = self.pool_layer(x_4)
        #
        x = self.conv5(x)
        x_6 = self.conv6(x)
        x = self.pool_layer(x_6)
        #
        x = self.conv7(x)
        x_8 = self.conv8(x)
        x = self.pool_layer(x_8)
        #
        x = self.conv9(x)
        x = self.conv10(x)
        x_11 = self.conv11(x)
        merge_11 = tf.concat(values=[x_8, x_11], axis = -1)
        x = self.conv12(merge_11)
        x = self.conv13(x)
        #
        x_14 = self.conv14(x)
        merge_14 = tf.concat([x_6, x_14], axis=-1)
        x = self.conv15(merge_14)
        x = self.conv16(x)
        #
        x_17 = self.conv17(x)
        merge_17 = tf.concat([x_4, x_17], axis=-1)
        x = self.conv18(merge_17)
        x = self.conv19(x)
        #
        x_20 = self.conv20(x)
        merge_20 = tf.concat([x_2, x_20], axis=-1)
        x = self.conv21(merge_20)
        x = self.conv22(x)
        x = self.conv23(x)
        return x

    def pool_layer(self, input, padding='SAME'):
        """
        """
        return tf.keras.layers.MaxPool2D(pool_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

    def un_conv(self, input, num_input_channels, conv_filter_size, num_filters, feature_map_size, train=True,
                padding='SAME', relu=True):
        """
        """
        init_biases = tf.keras.initializers.Constant(value=0.05)
        init_weights = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05)
        layer = tf.keras.layers.Conv2DTranspose(num_filters, conv_filter_size, strides=[1, 2, 2, 1], padding=padding,
                                                output_padding=None, kernel_initializer=init_weights,
                                                bias_initializer=init_biases)
#        layer = tf.nn.conv2d_transpose(input, filters=weights, output_shape=[batch_size_0, feature_map_size,
#                                                                                  feature_map_size, num_filters],
#                                       strides=[1, 2, 2, 1], padding=padding)
        if relu:
            layer = tf.keras.layers.ReLU(layer)
        return layer

    def conv_layer(self, input, num_input_channels, conv_filter_size, num_filters, padding='SAME', relu=True):
        """
        """
        init_biases = tf.keras.initializers.Constant(value=0.05)
        init_weights = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05)
        layer = tf.keras.layers.Conv2D(num_filters, conv_filter_size, strides=[1, 1, 1, 1], padding=padding,
                                       activation=None, use_bias=True, kernel_initializer=init_weights,
                                       bias_initializer=init_biases)
        if relu:
            layer = tf.keras.layers.ReLU(layer)
        return layer

    def create_unet(self, input_shape):
        """
        """
        self.conv1 = self.conv_layer(input_shape, 3, 3, 64)
        self.conv2 = self.conv_layer(64, 3, 64)
        self.pool2 = self.pool_layer()
        #
        self.conv3 = self.conv_layer(64, 3, 128)
        self.conv4 = self.conv_layer(128, 3, 128)
        self.pool4 = self.pool_layer()
        #
        self.conv5 = self.conv_layer(128, 3, 256)
        self.conv6 = self.conv_layer(256, 3, 256)
        self.pool6 = self.pool_layer()
        #
        self.conv7 = self.conv_layer(256, 3, 512)
        self.conv8 = self.conv_layer(512, 3, 512)
        self.pool8 = self.pool_layer()
        #
        self.conv9 = self.conv_layer(512, 3, 1024)
        self.conv10 = self.conv_layer(1024, 3, 1024)
        #
        # self.conv11 = self.un_conv(, 1024, 2, 512, self.img_size // 8 , train)
        self.conv11 = self.un_conv(1024, 2, 512)
        
        self.conv12 = self.conv_layer(1024, 3, 512)
        self.conv13 = self.conv_layer(512, 3, 512)
        #
        # self.conv14 = self.un_conv(512, 2, 256, self.img_size // 4 , train)
        self.conv14 = self.un_conv(512, 2, 256)

        self.conv15 = self.conv_layer(512, 3, 256)
        self.conv16 = self.conv_layer(256, 3, 256)
        #
        # self.conv17 = self.un_conv(256, 2, 128, self.img_size // 2,train)
        self.conv17 = self.un_conv(256, 2, 128)

        self.conv18 = self.conv_layer(256, 3, 128)
        self.conv19 = self.conv_layer(128, 3, 128)
        #
        self.conv20 = self.un_conv(128, 2, 64)

        self.conv21 = self.conv_layer(128, 3, 64)
        self.conv22 = self.conv_layer(64, 3, 64)
        self.conv23 = self.conv_layer(64, 1, self.output_classes, relu=False)
