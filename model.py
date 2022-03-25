"""
Colorize GAN architecture.
@author: Adrian Kucharski
"""
import datetime
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers
from skimage import color, io, morphology
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Dense, Dropout,
                                     Flatten, Input, LeakyReLU, MaxPool2D, Conv2DTranspose,
                                     UpSampling2D, AveragePooling2D, Reshape, GaussianNoise)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tqdm import tqdm
from typing import Union, Tuple
from glob import glob
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
from hexgrid import generate_hexagons

np.set_printoptions(suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_alizarine_dataset(path: str, mask_dilation: int = None) -> Tuple[np.ndarray]:
    'Returns Alizerine dataset with format [np.ndarray as {image, gt, roi}]'
    dataset = []
    for image_path in glob(os.path.join(path, 'images/*')):
        gt_path = image_path.replace('images', 'gt')
        roi_path = image_path.replace('images', 'roi')

        image = (io.imread(image_path, as_gray=True)[
            np.newaxis, ..., np.newaxis] - 127.5) / 127.5
        gt = io.imread(gt_path, as_gray=True) / 255.0
        roi = io.imread(roi_path, as_gray=True)[
            np.newaxis, ..., np.newaxis] / 255.0

        if mask_dilation is not None:
            gt = morphology.dilation(gt, np.ones(
                (mask_dilation, mask_dilation)))
        gt = gt[np.newaxis, ..., np.newaxis]
        dataset.append(np.concatenate([image, gt, roi], axis=0))

    return dataset

class HexagonDataIterator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=32, patch_size=64, total_patches=768 * 30, noise_size=(64,)):
        """Initialization
        Dataset is (x, y, roi)"""
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.total_patches = total_patches
        self.noise_size = noise_size
        self.on_epoch_end()

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return len(self.h) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        'Generate one batch of data'
        # Generate indexes of the batch
        idx = np.s_[index * self.batch_size:(index+1)*self.batch_size]
        h = self.h[idx]
        z = self.z[idx]
        return h, z

    def on_epoch_end(self):
        'Generate new hexagons after one epoch'
        self.h = generate_hexagons(self.total_patches,
                                   (17, 21), 0.65, random_shift=8)
        self.z = np.random.normal(0, 1, (self.total_patches, *self.noise_size))


class DataIterator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset: Tuple[np.ndarray], batch_size=32, patch_size=64, patch_per_image=768):
        """Initialization
        Dataset is (x, y, roi)"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_per_image = patch_per_image
        self.on_epoch_end()

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return len(self.x) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        'Generate one batch of data'
        # Generate indexes of the batch
        idx = np.s_[index * self.batch_size:(index+1)*self.batch_size]
        x = self.x[idx]
        y = self.y[idx]
        return y, x  # mask, image

    def _get_constrain_roi(self, roi: np.ndarray) -> Tuple[int, int, int, int]:
        'Get a posible patch position based on ROI'
        px, py, _ = np.where(roi != 0)
        pxy = np.dstack((px, py))[0]
        ymin, xmin = np.min(pxy, axis=0)
        ymax, xmax = np.max(pxy, axis=0)
        return [ymin, xmin, ymax, xmax]

    def on_epoch_end(self):
        'Generate new patches after one epoch'
        self.x, self.y = [], []
        mid = self.patch_size // 2
        for x, y, roi in self.dataset:
            ymin, xmin, ymax, xmax = self._get_constrain_roi(roi)
            xrand = np.random.randint(
                xmin + mid, xmax - mid, self.patch_per_image)
            yrand = np.random.randint(
                ymin + mid, ymax - mid, self.patch_per_image)

            for xpos, ypos in zip(xrand, yrand):
                self.x.append(x[ypos-mid:ypos+mid, xpos-mid:xpos+mid])
                self.y.append(y[ypos-mid:ypos+mid, xpos-mid:xpos+mid])

        self.x, self.y = np.array(self.x), np.array(self.y)

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
            optimizer=Adam(1e-4),
            loss='mae',
        )

        self.d_model = self._discriminator_model()
        self.d_model.compile(
            optimizer=Adam(2e-4, beta_1=0.5, clipnorm=1e-3),
            loss=BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.gan = self._gan_model()
        self.gan.compile(
            optimizer=Adam(1e-3, beta_1=0.5),
            loss=BinaryCrossentropy(from_logits=True),
        )
        self._create_dirs()
        self.writer = tf.summary.create_file_writer(self.log_path)

    def summary(self):
        self.d_model.summary()
        self.g_model.summary()
        self.gan.summary()

    def save_models(self, g_path: str = None, d_path: str = None):
        if g_path:
            path = os.path.join(self.g_path_save, g_path)
            self.g_model.save(path)
        if d_path:
            path = os.path.join(self.d_path_save, d_path)
            self.d_model.save(path)

    def write_log(self, names, metrics):
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
                metrics = self.d_model.train_on_batch(
                        [gts_join, images_join], labels_join)

                # Store discriminator metrics
                if step % log_per_steps == log_per_steps - 1:
                    tf.summary.experimental.set_step(epoch * steps + step)
                    self.write_log(d_names, metrics)

                # Train generator directly
                zt = tf.random.normal((len(gts), *self.noise_size))
                metrics_g = self.g_model.train_on_batch([gts, zt], images_real)

                # Train generator via discriminator
                metrics_gan = self.gan.train_on_batch([h, z], real_labels)

                # Store generator metrics
                if step % log_per_steps == log_per_steps - 1:
                    self.write_log(gan_names, [metrics_gan])
                    self.write_log(g_names, [metrics_g])

            self.save_models('model_last.h5', 'model_last.h5')
            if (epoch + 1) % save_per_epochs == 0:
                self._evaluate(epoch=epoch, data=val_data)
                self.save_models(f'model_{epoch}.h5')

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


if __name__ == '__main__':
    gan = GAN(patch_per_image=500)
    # gan.summary()
    datasetAlizarine = load_alizarine_dataset(
         'datasets/fold_1/', mask_dilation=None)
    gan.train(75, datasetAlizarine, save_per_epochs=1)


    # data_it = DataIterator(datasetAlizarine)
    # for x, y in data_it:
    #     xy = np.concatenate([x[0],y[0]], axis=1)
    #     plt.imshow(xy, cmap='gray')
    #     plt.show()

    # exit()
    # data = DataIterator(d)
    # for x, y in data:
    #     plt.imshow(np.concatenate([x[0], y[0]], axis=1), cmap='gray')
    #     plt.show()
    # hexagogData = HexagonDataIterator(total_patches=200)
    # for d in hexagogData:
    #     h, z = d
    #     plt.imshow(np.concatenate([h[0], z[0]], axis=1), cmap='gray')
    #     plt.show()
    # pass
