# -*- coding: utf-8 -*-

####
# JOUFFROY Emma 
####

# @TODO:  finish loss implementation, be careful of logs and output probabilities of images / mask,  
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time, datetime
from models.modelv2.Unetv2 import Unet

tfd = tfp.distributions

class Training():
    """
    This class computes the training for the variational autoencoder
    """
    def __init__(self, vae, unet, optimizer, batch_size, rd_vec, batch_max_train, batch_max_test, K, gamma):
        """
        model : instance of the class "variationalAutoEncoder"
        optimizer : Adam optimizer to perform sgd
        batch_size : number of images in one batch
        rd_vec : same batch of test images used to see improvement of the model 
        batch_max_train : number of batch in the train_dataset for one epoch
        batch_max_test : number of batch in the test_dataset for one epoch
        """
        self.vae = vae
        self.unet = unet
        self.K = K
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.rd_vec = rd_vec
        self.batch_max_train = batch_max_train
        self.batch_max_test = batch_max_test
        self.gamma = gamma 

    @tf.function
    def compute_loss(self, batch):
        """
        batch : batch of images on which we compute the loss
        """
        for k in range(K):
            scope = tf.ones()
            decoder_likelihoods = []
            approx_posteriors = []
            if (k == K-1):
                # dans le cas où c'est le dernier K, on ne rentre pas dans
                # unet, mais le mask est égal au dernier scope, ce qui permet
                # de traiter toute l'image
                mask = scope
            else: 
                # dans le reste des cas, on retrouve bien le mask et le scope
                # grâce au réseau attentionnel
                first_mask = self.unet(tf.concat([batch, scope], axis=1))
                mask = scope * first_mask
                scope = tf.math.log(scope * (1-first_mask))
            # approx_posterior = p(z|x)
            approx_posterior = self.model.encoder(tf.concat([batch, mask], axis=1))
            approx_posteriors.append(approx_posterior)
            approx_posterior_sample = approx_posterior.sample()
            # decoder_likelihood = p(x|z)
            decoder_likelihood = mask * self.model.decoder(approx_posterior_sample)
            decoder_likelihoods.append(decoder_likelihood)        
        summed_likelihood = tf.reduce_sum(decoder_likelihoods)
        distortion = -tf.math.log(summed_likelihood)
        factorised_posteriors = tf.reduce_prod(approx_posteriors)
        latent_prior = self.model.make_mixture_prior()
        rate_img = self.model.beta * tfd.kl_divergence(factorised_posteriors, latent_prior)
        rate_mask = self.gamma * tfd.kl_divergence(mask, latent_prior)
        loss = distortion + rate_img + rate_mask
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

    def generate_and_save_images(self, epoch, test_input):
        """
        Generate images using the decoder, show them and save them
        in order to visualize the training

        epoch : the epoch number 
        test_input : batch of images used for visualisation
        """
        # returns the distribution of "test_input" decoded
        decoder_likelihood = self.model.sample(test_input)
        # getting the mean of this distribution
        predictions = decoder_likelihood.mean()
        # plot the generated images
        fig = plt.figure(figsize=(10,10))
        fig.suptitle('Batch of generated images', fontsize=16)
        for i in range(12):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
        # save the image in order to create a gif file later
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

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
            # we reset yhe keras metrics to zero
            train_loss.reset_states()
            test_loss.reset_states()
