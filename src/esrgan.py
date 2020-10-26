"""
Useful links:
https://www.youtube.com/watch?v=KULkSwLk62I "How Super Resolution Works"
https://medium.com/analytics-vidhya/esrgan-enhanced-super-resolution-generative-adversarial-network-using-keras-a34134b72b77

github repos:
https://github.com/fenghansen/ESRGAN-Keras/blob/master/ESRGAN.py
https://github.com/SavaStevanovic/ESRGAN/blob/master/models/ESRGAN.py
"""

import datetime
import os
import pickle

from data_loader import DataLoader

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Input, Lambda, LeakyReLU, PReLU, UpSampling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

import numpy as np
from PIL import Image


PRETRAINED_GENERATOR_WEIGHTS = "../model/weights/pretrained_generator.h5"
GENERATOR_MODEL = "../model/image_generator_model.h5"

PRETRAINED_GENERATOR_CHECKPOINT_DIR = "../model/checkpoints/pre_train"
FINE_TUNE_CHECKPOINT_DIR = "../model/checkpoints/fine_tune"

IMG_DIR_VAL_LR = "/Users/karin/programming/data/ortho-images/default_test_images/lr"
IMG_DIR_PREDICTED = "../test_predictions"

HR_CROPPED_IMG_SIZE = 96
SCALE_FACTOR = 4

PERCENT_OF_TRAINING_SET=0.5  # check out actual training set (size) and if everything would fit into memory

LEARNING_RATE = 1e-4


# **************************************
#
# **************************************
class Utils():
    # ***************
    # inference
    # ***************
    def test_predict(self, model, data_loader, lr_dir_path, trained_steps, sub_dir):
        ''' Create prediction example of selected epoch '''
        def create_img_dir(trained_steps):
            ''' My default test dump '''
            save_img_dir = f"{IMG_DIR_PREDICTED}/{sub_dir}/{trained_steps}"
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
            return save_img_dir
        
        def resolve_single_image(lr_file_path):
            lr_img = data_loader.load_single_image(lr_file_path, size=100)
            generated_hr = model.generator.predict(lr_img)
            generated_hr = 0.5 * generated_hr + 0.5
            return Image.fromarray((np.uint8(generated_hr*255)[0])) 

        save_img_dir = create_img_dir(trained_steps)
        file_names = os.listdir(lr_dir_path)
        
        for i, file_name in enumerate(file_names[:5]):  # here: 5 prediction examples
            lr_file_path = f"{lr_dir_path}/{file_name}"
            img = resolve_single_image(lr_file_path)
            img.save(f"{save_img_dir}/{file_name}")


# **************************************
#
# **************************************
class ESRGANModel(tf.Module):  # regarding parameter: https://stackoverflow.com/a/60509193
    def __init__(self, hr_shape, lr_shape, channels, upscaling_factor):
        self.channels = channels
        self.hr_shape = hr_shape
        self.lr_shape = lr_shape

        self.upscaling_factor = upscaling_factor

        # Scaling of losses
        self.loss_weights = {'percept': 1e-3, 'gen': 5e-3, 'pixel': 1e-2}

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.vgg = self.build_vgg()


    def build_vgg(self):
        vgg = VGG19(
            weights='imagenet', 
            input_shape=self.hr_shape, 
            include_top=False
        )
        output_layer = 20

        model = Model(vgg.input, vgg.layers[output_layer].output)
        model.trainable = False
    
        return model


    def build_generator(self, filters=64, n_dense_block=16, n_sub_block=2):
        def dense_block(x_in):
            b1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x_in)
            b1 = LeakyReLU(0.2)(b1)
            b1 = Concatenate()([x_in, b1])
            
            b2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b1)
            b2 = LeakyReLU(0.2)(b2)
            b2 = Concatenate()([x_in, b1, b2]) 
            
            b3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b2)
            b3 = LeakyReLU(0.2)(b3)
            b3 = Concatenate()([x_in, b1, b2, b3])
            
            b4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b3)
            b4 = LeakyReLU(0.2)(b4)
            b4 = Concatenate()([x_in, b1, b2, b3, b4])
            
            b5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(b4)
            b5 = Lambda(lambda x:x*0.2)(b5)
            b5 = Add()([b5, x_in])
            
            return b5


        def RRDB(x_in):
            x = dense_block(x_in)
            x = dense_block(x)
            x = dense_block(x)
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, x_in])
            return out


        def SubpixelConv2D(scale=2):
            def subpixel_shape(input_shape):
                dims = [
                    input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))
                ]
                output_shape = tuple(dims)
                return output_shape

            def subpixel(x):
                return tf.nn.depth_to_space(x, scale)

            return Lambda(subpixel, output_shape=subpixel_shape)


        def upsample(x, idx):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name=f'upSampleConv2d_{idx}')(x)
            x = SubpixelConv2D(2)(x)
            x = PReLU(shared_axes=[1, 2], name=f'upSamplePReLU_{idx}')(x)
            
            return x

        
        lr_input = Input(shape=(None, None, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=3, strides=1, padding='same')(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])

        # Upsampling depending on factor
        x = upsample(x, 1)
        x = upsample(x, 2)  # self.upscaling_factor > 2
        
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return Model(inputs=lr_input, outputs=hr_output)

        
    def build_discriminator(self, num_filters=64):
        def discriminator_block(x_in, filters, strides=1, batch_normalization=True):
            x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x_in)
            x = LeakyReLU(alpha=0.2)(x)
            if batch_normalization:
                x = BatchNormalization(momentum=0.8)(x)
            return x

        x_in = Input(shape=self.hr_shape)

        x = discriminator_block(x_in, num_filters, batch_normalization=False)
        x = discriminator_block(x, num_filters, strides=2)
        
        x = discriminator_block(x, num_filters*2)
        x = discriminator_block(x, num_filters*2, strides=2)
        
        x = discriminator_block(x, num_filters*4)
        x = discriminator_block(x, num_filters*4, strides=2)
        
        x = discriminator_block(x, num_filters*8)
        x = discriminator_block(x, num_filters*8, strides=2)

        x = Dense(num_filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1)(x)

        return Model(x_in, x)


# **************************************
#
# **************************************
class Pretrainer():
    def __init__(self):
        # ***
        # Input shape
        # ***
        self.channels = 3

        self.hr_height = self.hr_width = HR_CROPPED_IMG_SIZE  # assuming same dimensions for height and width
        self.lr_height = self.lr_width = HR_CROPPED_IMG_SIZE//SCALE_FACTOR
        
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # ***
        # data loader and utils
        # ***
        self.data_loader = DataLoader(HR_CROPPED_IMG_SIZE, SCALE_FACTOR, PERCENT_OF_TRAINING_SET)
        self.utils = Utils()

        # ***
        # model, loss and optimizer
        # ***
        self.loss = MeanSquaredError()
        
        # ***
        # save training in checkpoint
        # necessary to keep all values (i.e. optimizer) when interrupting & resuming training
        # ***
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer=Adam(learning_rate=LEARNING_RATE),
            model=ESRGANModel(self.hr_shape, self.lr_shape, self.channels, SCALE_FACTOR)
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=PRETRAINED_GENERATOR_CHECKPOINT_DIR,
            max_to_keep=1
        )

        self.restore_checkpoint()

    
    # ***
    # get the latest checkpoint (if existing)
    # ***
    def restore_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restore checkpoint at step {self.checkpoint.step.numpy()}.")


    # ***************
    # train model (generator only)
    # ***************
    @tf.function
    def pretrain_step(self, lr_img, hr_img):
        with tf.GradientTape() as tape:
            hr_generated = self.checkpoint.model.generator(lr_img, training=True)
            loss = self.loss(hr_img, hr_generated)

        gradients = tape.gradient(loss, self.checkpoint.model.generator.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.generator.trainable_variables))

        return loss


    def pretrain(self, epochs=10, batch_size=1, sample_interval=2):
        start_time = datetime.datetime.now()  # for controle dump only

        for epoch in range(epochs):
            self.checkpoint.step.assign_add(batch_size)  # update steps in checkpoint
            trained_steps = self.checkpoint.step.numpy() # overall trained steps: for controle dump only
            
            # ***
            # train on batch
            # ***
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size)
            loss = self.pretrain_step(imgs_lr, imgs_hr)

            elapsed_time = datetime.datetime.now() - start_time  # for controle dump/log only

            # ***
            # save and/or dump
            # ***
            if (epoch + 1) % 10 == 0:
                print(f"{epoch + 1} | steps: {trained_steps} | loss: {loss} | time: {elapsed_time}")
            if (epoch + 1) % sample_interval == 0:
                print("   |---> save and make image sample")
                self.checkpoint_manager.save()  # save checkpoint

                # save weights for usage with class Trainer on first training iteration
                self.checkpoint.model.generator.save_weights(PRETRAINED_GENERATOR_WEIGHTS)

                # controle dump of predicted images: save in dirs named by trained_steps
                self.utils.test_predict(self.checkpoint.model, self.data_loader, IMG_DIR_VAL_LR, trained_steps, "pre_train")


# **************************************
#
# **************************************
class Trainer():
    def __init__(self, use_pretrain_weights=False):
        # ***
        # Input shape
        # ***
        self.channels = 3

        self.hr_height = self.hr_width = HR_CROPPED_IMG_SIZE  # assuming same dimensions for height and width
        self.lr_height = self.lr_width = HR_CROPPED_IMG_SIZE//SCALE_FACTOR
        
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # ***
        # data loader and utils
        # ***
        self.data_loader = DataLoader(HR_CROPPED_IMG_SIZE, SCALE_FACTOR, PERCENT_OF_TRAINING_SET)
        self.utils = Utils()

        # ***
        # loss
        # ***
        self.loss_binary_crossentropy = BinaryCrossentropy(from_logits=True)  # True or False? https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
        self.loss_mse = MeanSquaredError()

        # ***
        # save training in checkpoint
        # necessary to keep all values when interrupting & resuming training
        # ***
        
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer_generator=Adam(learning_rate=LEARNING_RATE),
            optimizer_discriminator=Adam(learning_rate=LEARNING_RATE),
            model=ESRGANModel(self.hr_shape, self.lr_shape, self.channels, SCALE_FACTOR)
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=FINE_TUNE_CHECKPOINT_DIR,
            max_to_keep=1
        )

        # ***
        # either: start training with pretrained generator weights a) if parameter is True and b) if weights are available
        # or: get latest checkpoint of training  a) if parameter is False and b) if checkpoint is available
        # ***
        if use_pretrain_weights is False:
            self.restore_checkpoint()
        else:
            try:
                self.checkpoint.model.generator.load_weights(PRETRAINED_GENERATOR_WEIGHTS)
                print(f"Load pretrained generator weights.")
            except:
               print("No pre-trained weights available.")
               pass

    
    # ***
    # get the latest checkpoint (if existing)
    # ***
    def restore_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restore checkpoint at step {self.checkpoint.step.numpy()}.")

    # ***
    # loss functions, used in training step
    # ***
    @tf.function
    def content_loss(self, hr_img, hr_generated):
        def preprocess_vgg(x):
            '''
            Input img RGB [-1, 1] -> BGR [0,255] plus subtracting mean BGR values: (103.939, 116.779, 123.68)
            https://stackoverflow.com/a/46623958
            '''
            if isinstance(x, np.ndarray):
                return preprocess_input((x+1)*127.5)
            else:            
                return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5), autocast=False)(x)  # autocast: tf 2.something / float32 vs. float64 with Lambda
        
        hr_generated = preprocess_vgg(hr_generated)
        hr_img = preprocess_vgg(hr_img)

        # ***
        # VGG scaled with 1/12.75 as in paper
        # Compare with: http://krasserm.github.io/2019/09/04/super-resolution/ (train.py)
        # ***
        hr_generated_features = self.checkpoint.model.vgg(hr_generated)/12.75 
        hr_features = self.checkpoint.model.vgg(hr_img)/12.75 

        return self.loss_mse(hr_features, hr_generated_features)


    def generator_loss(self, real_logit, fake_logit):
        generator_loss = K.mean(
            K.binary_crossentropy(K.zeros_like(real_logit), real_logit) + 
            K.binary_crossentropy(K.ones_like(fake_logit), fake_logit)
        )

        return generator_loss


    def discriminator_loss(self, fake_logit, real_logit):
        discriminator_loss = K.mean(
            K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
            K.binary_crossentropy(K.ones_like(real_logit), real_logit)
        )
        
        return discriminator_loss


    def relativistic_loss(self, real_img, fake_img):
        fake_logit = K.sigmoid(fake_img - K.mean(real_img))
        real_logit = K.sigmoid(real_img - K.mean(fake_img))

        return fake_logit, real_logit


    # ***
    #
    # ***
    @tf.function
    def train_step(self, lr_img, hr_img):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            hr_generated = self.checkpoint.model.generator(lr_img, training=True)

            # ***
            # discriminator
            # ***
            hr_output = self.checkpoint.model.discriminator(hr_img, training=True)
            hr_generated_output = self.checkpoint.model.discriminator(hr_generated, training=True)

            fake_logit, real_logit = self.relativistic_loss(hr_output, hr_generated_output)

            discriminator_loss = self.discriminator_loss(fake_logit, real_logit)

            # ***
            # generator
            # ***
            content_loss = self.content_loss(hr_img, hr_generated)
            generator_loss = self.generator_loss(real_logit, fake_logit)
            perceptual_loss = content_loss + 0.001 * generator_loss
            

        gradients_generator = gen_tape.gradient(perceptual_loss, self.checkpoint.model.generator.trainable_variables)
        gradients_discriminator = disc_tape.gradient(discriminator_loss, self.checkpoint.model.discriminator.trainable_variables)

        self.checkpoint.optimizer_generator.apply_gradients(zip(gradients_generator, self.checkpoint.model.generator.trainable_variables))
        self.checkpoint.optimizer_discriminator.apply_gradients(zip(gradients_discriminator, self.checkpoint.model.discriminator.trainable_variables))

        return perceptual_loss, discriminator_loss

    
    # ***
    #
    # ***
    def train(self, epochs=10, batch_size=1, sample_interval=2):
        start_time = datetime.datetime.now()  # for controle dump only

        for epoch in range(epochs):
            self.checkpoint.step.assign_add(batch_size)  # update steps in checkpoint
            trained_steps = self.checkpoint.step.numpy() # overall trained steps: for controle dump only
            
            # ***
            # train on batch
            # ***
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size)
            perceptual_loss, discriminator_loss = self.train_step(imgs_lr, imgs_hr)

            elapsed_time = datetime.datetime.now() - start_time  # for controle dump only

            # ***
            # save and/or dump
             # ***
            if (epoch + 1) % 10 == 0:
                print(f"{epoch + 1} | steps: {trained_steps} | g_loss: {perceptual_loss} | d_loss: {discriminator_loss} | time: {elapsed_time}")
            if (epoch + 1) % sample_interval == 0:
                print("   |---> save and make image sample")
                self.checkpoint_manager.save()  # save checkpoint
                self.checkpoint.model.generator.save(GENERATOR_MODEL)  # save complete model for actual usage

                # controle dump of predicted images: save in dirs named by trained_steps
                self.utils.test_predict(self.checkpoint.model, self.data_loader, IMG_DIR_VAL_LR, trained_steps, "fine_tune")
        


if __name__ == '__main__':
    # ***
    # 1. pre-train generator (only)
    # ***
    pretrainer = Pretrainer()
    pretrainer.pretrain(epochs=10, batch_size=1, sample_interval=5)

    # ***
    # 2. train generator and discriminator
    # ***
    # trainer = Trainer(use_pretrain_weights=True)  # use this parameter on very first training run (default: False)
    # trainer = Trainer()  # use this if you continue training (i.e. after interruption)
    # trainer.train(epochs=4000, batch_size=4, sample_interval=1000)
