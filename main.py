from tensorflow.keras import Input
from data.load_data import load_dataset
from models.unet.Unet import Unet
import tensorflow as tf
# from models.modelv2.Unetv2 import Unet


if __name__ == "__main__":
    filename = '/Users/jouffroy/Desktop/thèse/UNET_repro/data/images.tfrecords'
    img_size = (240, 1240, 1)
    input_img = Input(shape=img_size, name='input_image')
    input_scope = Input(shape=img_size, name='input_scope')
    input_net = tf.concat((input_img, input_scope), axis=3)

    batch_size = 1
    num_classes = 11
    ds = load_dataset(filename, batch_size)
    batch = next(iter(ds))
    unet = Unet(num_classes, img_size, batch_size, input_img, input_scope)
    # print(pred.shape)
    # epoch = 30
    # optimizer = tf.keras.optimizers.SGD(1e-3, 0.99)
    # train_size = 12
    # test_size = 12
    # batch_max_train = np.array(tf.floor(train_size/batch_size))
    # batch_max_test = np.array(tf.math.ceil(test_size/batch_size))
    # training = Training(unet, optimizer, batch_size, batch_max_train, batch_max_test,11)
    # training.train(ds, ds, 2)

