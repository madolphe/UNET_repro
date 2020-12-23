from tensorflow.keras import Input
from data.load_data import load_dataset
import tensorflow as tf
from models.modelv2.Unetv2 import Unet
import numpy as np
from train.train import Training
from train.sanity_check import Sanity_Check

if __name__ == "__main__":
    # @TODO: relative path + majuscule for hyperparameteres + split dataset train / test
    filename = '/Users/jouffroy/Desktop/thèse/UNET_repro/data/images.tfrecords'
    img_size = (240, 240, 3)

    batch_size = 1
    num_classes = 2
    epoch = 30
    train_size = 12
    test_size = 12
    batch_max_train = np.array(tf.floor(train_size/batch_size))
    batch_max_test = np.array(tf.math.ceil(test_size/batch_size))

    ds = load_dataset(filename, batch_size)
    unet = Unet(img_size)
    optimizer = tf.keras.optimizers.SGD(1e-3, 0.99)    

    # sc = Sanity_Check(ds=ds, model=unet.model, training='osef')
    # sc.check_output_model()
    
    training = Training(unet.model, optimizer, batch_size, batch_max_train, batch_max_test, num_classes)
    training.train(ds, ds, epoch)
    print("OK TOUT VAS BIEN ON A TERMINE :D ")

