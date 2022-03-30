"""
Colorize GAN architecture.
@author: Adrian Kucharski
"""
import datetime
import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from skimage import io
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization, GaussianNoise, GaussianDropout,
                                     Concatenate, Conv2D, Conv2DTranspose, UpSampling2D,
                                     Dense, Dropout, Flatten, Input, LeakyReLU,
                                     MaxPool2D)
from tensorflow.keras.losses import BinaryCrossentropy, Hinge
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from dataset import DataIterator, HexagonDataIterator

np.set_printoptions(suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GAN():
    def __init__(self,
                 g_path_save='generator/models',
                 d_path_save='discriminator/models',
                 evaluate_path_save='data/images/',
                 log_path='logs/gan/',
                 patch_size=64,
                 patch_per_image=768,
                 g_path_last_name_save='model_last.h5',
                 d_path_last_name_save='model_last.h5'
                 ):
        self.log_path = log_path
        self.g_path_save = g_path_save
        self.d_path_save = d_path_save
        self.evaluate_path_save = evaluate_path_save
        self.patch_size = patch_size
        self.patch_per_image = patch_per_image
        self.g_path_last_name_save = g_path_last_name_save
        self.d_path_last_name_save = d_path_last_name_save

        self.noise_size = (patch_size, patch_size, 1)
        self.input_size = (patch_size, patch_size, 1)
        self.input_disc_size = (patch_size, patch_size, 1)

        self.g_model = self._generator_model()
        self.g_model.compile(
            optimizer=Adam(1e-5),
            loss='mae',
        )

        self.d_model = self._discriminator_model()
        self.d_model.compile(
            optimizer=Adam(2e-4, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Disable a discriminator training during gan training
        self.d_model.trainable = False

        self.gan = self._gan_model()
        self.gan.compile(
            optimizer=Adam(2e-4, beta_1=0.5),
            loss='binary_crossentropy',
        )
        self._create_dirs()
        self.writer = tf.summary.create_file_writer(self.log_path)
        self.gan_log_names = ['gan_loss']
        self.d_log_names = ['d_loss', 'd_acc']
        self.g_log_names = ['g_loss']

    def _evaluate(self, epoch: int, data: Tuple[np.ndarray, np.ndarray]=None) -> Union[None, Tuple[np.ndarray]]:
        if data is not None:
            images = []
            path = os.path.join(self.evaluate_path_save, str(epoch))
            Path(path).mkdir(parents=True, exist_ok=True)

            xdata, ydata = data
            z = np.random.normal(size=(len(xdata), *self.noise_size))
            pred = (self.g_model.predict_on_batch([xdata, z]) + 1) / 2.0

            for i in range(len(xdata)):
                x, y = xdata[i], ydata[i]
                impath = os.path.join(path, f'org_{i}.png')
                image = np.array(np.concatenate([x, pred[i], (y + 1) / 2.0], axis=1) * 255, 'uint8')
                io.imsave(impath, image)
                images.append(image)
            
            return np.array(images, dtype='uint8')
        return None

    def _save_models(self, g_path: str = None, d_path: str = None):
        if g_path:
            path = os.path.join(self.g_path_save, g_path)
            self.g_model.save(path)
        if d_path:
            path = os.path.join(self.d_path_save, d_path)
            self.d_model.save(path)

    def _write_log(self, names, metrics):
        with self.writer.as_default():
            for name, value in zip(names, metrics):
                tf.summary.scalar(name, value)
            self.writer.flush()
    
    def _write_images(self, epoch: int, images: np.ndarray):
        with self.writer.as_default():
            tf.summary.image("Validation data", images, step=epoch, max_outputs=len(images), description="Mask|Generated|Orginal")
        self.writer.flush()

    def _create_dirs(self):
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.log_path = os.path.join(self.log_path, time)
        self.g_path_save = os.path.join(self.g_path_save, time)
        self.d_path_save = os.path.join(self.d_path_save, time)
        self.evaluate_path_save = os.path.join(self.evaluate_path_save, time)
        for path in [self.log_path, self.g_path_save, self.d_path_save, self.evaluate_path_save]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _gan_model(self):
        H = h = Input(self.input_size, name='mask')
        Z = z = Input(self.noise_size, name='noise')
        generator_out = self.g_model([h, z])
        discriminator_out = self.d_model([h, generator_out])
        return Model(inputs=[H, Z], outputs=discriminator_out, name='GAN')

    def _discriminator_model(self):
        h = Input(self.input_disc_size, name='mask')
        t = Input(self.input_disc_size, name='image')

        inputs = Concatenate()([h, t])
        x = Conv2D(64, 5, padding='same')(inputs) 
        x = LeakyReLU(0.2)(x)
        x = MaxPool2D((2, 2))(x)

        x = Dropout(0.25)(x)

        # 32x32 @ 3x3 -> 30x30
        x = Conv2D(128, 3, padding='valid')(x)
        x = LeakyReLU(0.2)(BatchNormalization()(x))
        x = MaxPool2D((2, 2))(x)

        # 15x15 @ 3x3 -> 13x13
        x = Conv2D(256, 3, padding='valid')(x)
        x = LeakyReLU(0.2)(BatchNormalization()(x))

        # 13x13 @ 3x3 -> 11x11
        x = Conv2D(1, 3, padding='valid', activation='sigmoid')(x)
        return Model(inputs=[h, t], outputs=x, name='discriminator')

    def _generator_model(self):
        H = h = Input(self.input_size, name='mask')
        Z = z = Input(self.noise_size, name='noise')
        x = Concatenate()([h, z])

        encoder = []
        kernels = 3
        filters, fn, fm = [16, 32, 64], 64, 16
        for f in filters:
            x = Conv2D(f, kernels, padding='same', activation='relu')(x)
            encoder.append(x)
            x = MaxPool2D((2, 2))(x)

        x = Conv2D(fn, kernels, padding='same', activation='relu')(x)

        for f in filters[::-1]:
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(f, kernels, padding='same', activation='relu')(x)
            x = Concatenate()([encoder.pop(), x])
            
        x = GaussianDropout(0.1)(x, training=True)
        x = Conv2D(fm, kernels, padding='same', activation='relu')(x)
        outputs = Conv2D(1, 3, padding='same', activation='tanh', name='output')(x)
        return Model(inputs=[H, Z], outputs=outputs, name='generator')

    def train(self, epochs: int, dataset: Tuple[np.ndarray], evaluate_data: Tuple[np.ndarray] = None, batch_size=128, save_per_epochs=5, log_per_steps=5):
        # Prepare label arrays for D and GAN training
        real_labels = tf.ones(
            (batch_size, *self.d_model.output_shape[1:]), dtype=tf.float32)
        fake_labels = tf.zeros(
            (batch_size, *self.d_model.output_shape[1:]), dtype=tf.float32)
        labels_join = tf.concat([real_labels, fake_labels], axis=0)

        for epoch in tqdm(range(epochs)):
            # Init iterator
            data_it = DataIterator(
                dataset, batch_size, self.patch_size, self.patch_per_image, inv_values=True)
            data_hz = HexagonDataIterator(
                batch_size, self.patch_size, self.patch_per_image * len(dataset), self.noise_size, inv_values=True)
            steps = len(data_it)
            assert steps > log_per_steps

            # Training discriminator loop
            for step, ((gts, images_real), (h, z)) in enumerate(zip(data_it, data_hz)):
                # Concatenate fake with true
                image_fake = self.g_model.predict_on_batch([h, z])

                # Train discriminator on predicted and real and fake data
                gts_join = tf.concat([gts, h], axis=0)
                images_join = tf.concat([images_real, image_fake], axis=0)
                metrics_d = self.d_model.train_on_batch(
                    [gts_join, images_join], labels_join)

                # Train generator directly
                z = tf.random.normal((len(gts), *self.noise_size))
                metrics_g = self.g_model.train_on_batch([gts, z], images_real)

                # Train generator via discriminator
                z = tf.random.normal((len(h), *self.noise_size))
                metrics_gan = self.gan.train_on_batch([h, z], real_labels)

                # Store generator and discriminator metrics
                if step % log_per_steps == log_per_steps - 1:
                    tf.summary.experimental.set_step(epoch * steps + step)
                    self._write_log(self.gan_log_names, [metrics_gan])
                    self._write_log(self.g_log_names, [metrics_g])
                    self._write_log(self.d_log_names, metrics_d)

            self._save_models(self.g_path_last_name_save, self.d_path_last_name_save)
            if (epoch + 1) % save_per_epochs == 0:
                images = self._evaluate(epoch=epoch, data=evaluate_data)
                self._write_images(epoch, images)
                self._save_models(f'model_{epoch}.h5')

    def summary(self):
        self.d_model.summary()
        self.g_model.summary()
        self.gan.summary()
