
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time, datetime



class Training():
    """
    """
    def __init__(self, model, optimizer, batch_size, rd_vec, batch_max_train, batch_max_test):
        """
        model : instance of the class "variationalAutoEncoder"
        optimizer : Adam optimizer to perform sgd
        batch_size : number of images in one batch
        rd_vec : same batch of test images used to see improvement of the model 
        batch_max_train : number of batch in the train_dataset for one epoch
        batch_max_test : number of batch in the test_dataset for one epoch
        """
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.rd_vec = rd_vec
        self.batch_max_train = batch_max_train
        self.batch_max_test = batch_max_test

    @tf.function
    def compute_loss(self, batch):
        """
        """
        return loss

    @tf.function
    def compute_apply_gradients(self, batch):
        """
        Implements a training loop using eager mode. It records
        operations for automatic differenciation using SGD. 
        batch : batch of images on which we compute the loss
        """
        # records the operations to calcul the loss into the "tape" object
        with tf.GradientTape() as tape:
            # computes the loss for a batch
            loss = self.compute_loss(batch)
        # Computes the derivative of the loss with respect of each trainable variables
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # apply the processed gradients on each trainable variables of the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


    def train(self, train_dataset, test_dataset, epochs, rd_vec):
        """
        Trains the model for each epoch 
        train_dataset : dataset for training the model
        test_dataset : dataset for testing the model
        epochs : number of total epochs
        rd_vec : same batch of images used to see the improvment of the model
        """
        # current time to create logs folder
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        # creations of folders for test and train logs in order to see them
        # in tensorboard
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        # creation of keras mean metrics for train and test elbo loss 
        train_loss = tf.keras.metrics.Mean('train_loss')
        test_loss = tf.keras.metrics.Mean('test_loss')
        
        # loop on each epoch for training
        for epoch in range(1, epochs+1):
            # initialization of progress bar for visualization
            progress_bar_train = tf.keras.utils.Progbar(self.batch_max_train)
            progress_bar_test = tf.keras.utils.Progbar(self.batch_max_test)
            i=0
            j=0
            print('------------------------------------------------------')
            print('Epoch {} on {} : '.format(epoch, epochs))
            print('On training set:')

            # loop on each batch contained in training set
            for train_batch in train_dataset:
                # computing the loss and applying SGD on the batch selected
                loss_train = self.compute_apply_gradients(train_batch[0])
                i+=1
                # adding the loss on the keras metric 
                train_loss.update_state(loss_train)
                # updating the progress bar for visualization
                progress_bar_train.update(i)
                if (i >= self.batch_max_train):
                # we need to break the loop by hand because
                # the generator loops ndefinitely
                    break
            with train_summary_writer.as_default():
                # adding the result of the training loss in the summary
                # for visualization on tensorflow
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
            print('On testing set:')
            # loop on each batch of testing set
            for test_batch in test_dataset:
                # computing the loss without applying SGD
                loss_test = self.compute_loss(test_batch[0])
                j+=1
                # adding the loss on the keras metric
                test_loss.update_state(loss_test)
                # updating the progress bar for visualization
                progress_bar_test.update(j)
                if (j >= self.batch_max_test):
                # we need to break the loop by hand because
                # the generator loops ndefinitely
                    break
            with test_summary_writer.as_default():
                # adding the result of the testing loss in the summary
                #for visualization on tensorflow
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
            
            # print the resulting loss over each epoch for training set
            # et testing set
            template = 'Epoch {}, Loss: {}, Test Loss: {}'
            print (template.format(epoch,
                        train_loss.result(), 
                        test_loss.result()))
            # generate image of a batch of testing images for each epoch
            # to visualize the improvment of the network
            if(epoch % 20 ==0):
                self.model.save_weights('./checkpoints/my_checkpoint/beta1')
                print("model saved successfully for epoch", str(epoch))
            self.generate_and_save_images(epoch, self.rd_vec)
            # we reset the keras metrics to zero
            train_loss.reset_states()
            test_loss.reset_states()