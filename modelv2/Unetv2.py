import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from operator import floordiv 
from tensorflow.keras.initializers import TruncatedNormal, Zeros



class Unet(tf.keras.Model):
    def __init__(self, input_img, input_scope, name='unet', **kwargs):
        super(Unet, self).__init__(name=name, **kwargs)
        
        self.input_img = input_img
        self.input_scope = input_scope
        self.input_net = tf.concat((input_img, input_scope), axis=3)
        self.w_init = TruncatedNormal


        def down_block(prev_layer, filters, downsample=True):
            conv = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=w_init(), use_bias=False)(prev_layer)
            norm = InstanceNormalization()(conv)
            acti = ReLU()(norm)
            size = K.int_shape(acti)
        
            down = tf.image.resize(acti, (size[1]//2, size[2]//2), method='nearest')
            
            if downsample:
                return down, acti
            else:
                return acti

        def up_block(prev_layer, skip, filters, upsample=True):
            cat = tf.concat((prev_layer, skip), axis=3)
            conv = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=w_init(), use_bias=False)(cat)
            norm = InstanceNormalization()(conv)
            acti = ReLU()(norm)
            
            size = K.int_shape(acti)
        
            up = tf.image.resize(acti, (size[1]*2, size[2]*2), method='nearest')
            
            if upsample:
                return up
            else:
                return acti

    def call(self):
        d1, s1 = self.down_block(self.input_net, 64)
        d2, s2 = self.down_block(d1, 128)
        d3, s3 = self.down_block(d2, 256)
        d4, s4 = self.down_block(d3, 512)
        s5 = self.down_block(d4, 1024, False)
        flat = Flatten()(s5)
        fully1 = Dense(128, activation='relu')(flat)
        fully2 = Dense(128, activation='relu')(fully1)
        fully3 = Dense(K.int_shape(flat)[1], activation='relu')(fully2)
        out = tf.reshape(fully3, K.shape(s5))
        u1 = up_block(out, s5, 1024)
        u2 = up_block(u1, s4, 512)
        u3 = up_block(u2, s3, 256)
        u4 = up_block(u3, s2, 128)
        u5 = up_block(u4, s1, 64, False)
        output = Conv2D(2, (1, 1), strides=1, padding='same', kernel_initializer=w_init(), use_bias=False)(u5)
        return output
     