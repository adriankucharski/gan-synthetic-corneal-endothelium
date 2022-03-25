"""
Colorize GAN architecture.
@author: Adrian Kucharski
"""
import datetime
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage import io
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Dense, Dropout,
                                     Flatten, Input, LeakyReLU, Conv2DTranspose,)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tqdm import tqdm
from typing import Tuple
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
                 patch_per_image=768
                 ):
        self.log_path = log_path
        self.g_path_save = g_path_save
        self.d_path_save = d_path_save
        self.evaluate_path_save = evaluate_path_save
        self.patch_size = patch_size
        self.patch_per_image = patch_per_image

        self.noise_size = (patch_size, patch_size, 1)
        self.input_size = (patch_size, patch_size, 1)
        self.input_disc_size = (patch_size, patch_size, 1)

        self.g_model = self._generator_model()
        self.g_model.compile(
            optimizer=Adam(2e-4),
            loss='mae',
        )

        self.d_model = self._discriminator_model()
        self.d_model.compile(
            optimizer=Adam(2e-4, beta_1=0.5, clipnorm=1e-3),
            loss=BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Disable discriminator training during gan training
        self.d_model.trainable = False

        self.gan = self._gan_model()
        self.gan.compile(
            optimizer=Adam(1e-3, beta_1=0.5, clipnorm=1e-3),
            loss=BinaryCrossentropy(from_logits=True),
        )
        self._create_dirs()
        self.writer = tf.summary.create_file_writer(self.log_path)

    def _evaluate(self, epoch: int = None, num_of_test=8, data=None):
        np.random.seed(7312)
        data_hz = HexagonDataIterator(
            num_of_test, self.patch_size, num_of_test, self.noise_size)
        np.random.seed(None)
        h, z = data_hz.h, data_hz.z
        y = (self.g_model.predict_on_batch([h, z]) + 1) / 2.0

        path = self.evaluate_path_save
        if epoch is not None:
            path = os.path.join(self.evaluate_path_save, str(epoch))
            Path(path).mkdir(parents=True, exist_ok=True)

        for i in range(num_of_test):
            impath = os.path.join(path, f'{i}.png')
            io.imsave(impath, np.array(np.concatenate(
                [h[i], y[i]], axis=1) * 255, 'uint8'))

        if data is not None:
            z = np.random.normal(size=(len(data), *self.noise_size))
            xdata = data[:, 0, ...]
            pred = (self.g_model.predict_on_batch([xdata, z]) + 1) / 2.0
            for i in range(len(data)):
                x, y = data[i]
                impath = os.path.join(path, f'org_{i}.png')
                io.imsave(impath, np.array(np.concatenate(
                    [x, pred[i], (y + 1) / 2.0], axis=1) * 255, 'uint8'))

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

    def _create_dirs(self):
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.log_path = os.path.join(self.log_path, time)
        self.g_path_save = os.path.join(self.g_path_save, time)
        self.d_path_save = os.path.join(self.d_path_save, time)
        self.evaluate_path_save = os.path.join(self.evaluate_path_save, time)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        Path(self.g_path_save).mkdir(parents=True, exist_ok=True)
        Path(self.d_path_save).mkdir(parents=True, exist_ok=True)
        Path(self.evaluate_path_save).mkdir(parents=True, exist_ok=True)

    def _gan_model(self):
        H = h = Input(self.input_size, name='mask')
        Z = z = Input(self.noise_size, name='noise')
        generator_out = self.g_model([h, z])
        discriminator_out = self.d_model([h, generator_out])
        return Model(inputs=[H, Z], outputs=discriminator_out, name='GAN')

    def _discriminator_model(self):
        h = Input(self.input_disc_size, name='mask')
        t = Input(self.input_disc_size, name='image')
        i = RandomNormal(stddev=1e-1)
        inputs = Concatenate()([h, t])
        x = Conv2D(64, (5, 5), padding='same', kernel_initializer=i)(inputs)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(128, (5, 5), strides=(2, 2),
                   padding='same', kernel_initializer=i)(x)
        x = LeakyReLU(0.2)(BatchNormalization()(x))
        x = Dropout(0.5)(x)

        x = Conv2D(256, (5, 5), strides=(2, 2),
                   padding='same', kernel_initializer=i)(x)
        x = LeakyReLU(0.2)(BatchNormalization()(x))
        x = Dropout(0.5)(x)

        x = Conv2D(512, (5, 5), strides=(2, 2),
                   padding='same', kernel_initializer=i)(x)
        x = LeakyReLU(0.3)(BatchNormalization()(x))

        x = Flatten()(x)
        x = Dense(1)(x)
        return Model(inputs=[h, t], outputs=x, name='discriminator')

    def _generator_model(self):
        H = h = Input(self.input_size, name='mask')
        Z = z = Input(self.noise_size, name='noise')
        x = Concatenate()([h, z])
        i = RandomNormal(stddev=1e-1)

        def ConvBlock(filters, kernel=3, strides=1, activation='relu'):
            return Sequential([
                Conv2D(filters, kernel, strides=strides,
                       padding='same', kernel_initializer=i),
                BatchNormalization(),
                Activation(activation)
            ])

        encoder = []
        kernels = 3
        filters, n, m = [32, 64, 128], 128, 32
        for f in filters:
            x = ConvBlock(f, kernel=kernels, strides=(1, 1))(x)
            x = Dropout(0.5)(x)
            encoder.append(x)
            x = ConvBlock(f, kernel=kernels, strides=(2, 2))(x)

        x = ConvBlock(n, kernel=kernels)(x)

        for f in filters[::-1]:
            x = Conv2DTranspose(f, kernels, (2, 2),
                                padding='same', activation='relu')(x)
            x = ConvBlock(f, kernel=kernels)(x)
            x = Concatenate()([encoder.pop(), x])

        x = ConvBlock(m, kernels)(x)
        outputs = Conv2D(1, (3, 3), padding='same',
                         activation='tanh', name='output')(x)
        return Model(inputs=[H, Z], outputs=outputs, name='generator')

    def train(self, epochs: int, dataset: Tuple[np.ndarray], batch_size=128, save_per_epochs=5, log_per_steps=5):
        gan_names = ['gan_loss']
        d_names = ['d_loss', 'd_acc']
        g_names = ['g_loss']

        val_data = [[x[0], y[0]] for x, y in DataIterator(
            dataset, batch_size=1, patch_per_image=1)][0:8]
        val_data = np.array(val_data)

        for epoch in tqdm(range(epochs)):
            # Init iterator
            data_it = DataIterator(
                dataset, batch_size, self.patch_size, self.patch_per_image)
            data_hz = HexagonDataIterator(
                batch_size, self.patch_size, self.patch_per_image * len(dataset), self.noise_size)
            steps = len(data_it)
            assert steps > log_per_steps

            # real is for real images values
            # fake is for predicted pix2pix values
            real_labels = tf.ones((batch_size, 1), dtype=tf.float32)
            fake_labels = tf.zeros((batch_size, 1), dtype=tf.float32)
            labels_join = tf.concat([real_labels, fake_labels], axis=0)

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
                zt = tf.random.normal((len(gts), *self.noise_size))
                metrics_g = self.g_model.train_on_batch([gts, zt], images_real)

                # Train generator via discriminator
                metrics_gan = self.gan.train_on_batch([h, z], real_labels)

                # Store generator and discriminator metrics
                if step % log_per_steps == log_per_steps - 1:
                    tf.summary.experimental.set_step(epoch * steps + step)
                    self._write_log(gan_names, [metrics_gan])
                    self._write_log(g_names, [metrics_g])
                    self._write_log(d_names, metrics_d)

            self._save_models('model_last.h5', 'model_last.h5')
            if (epoch + 1) % save_per_epochs == 0:
                self._evaluate(epoch=epoch, data=val_data)
                self._save_models(f'model_{epoch}.h5')

    def summary(self):
        self.d_model.summary()
        self.g_model.summary()
        self.gan.summary()
