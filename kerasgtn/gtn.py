import numpy as np
import time


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop


class GTN(object):
    def __init__(self, real_input_shape, n_classes, noise_shape=(100,), optimizer=None):

        # shape of the real data without batch size
        self.input_shape = input_shape

        # number of classes
        self.n_classes = n_classes

        # shape of the noise vector for the generator
        self.noise_shape = noise_shape
        
        # optimizer
        self.optimizer = RMSprop() if optimizer is None else optimizer

        # populated with functions below
        self.learner = None
        self.generator = None
        self.model = None

    def get_learner(self):
        """
        :return: an uncompiled keras.models.Sequential model that represents a learner
        """
        if self.learner:
            return self.learner
        raise NotImplementedError

    def get_generator(self):
        """
        A template generator, feel free to overwrite to meet needs.
        :return: an uncompiled keras.models.Sequential model that represents a generator
        """
        if self.generator:
            return self.generator
        self.generator = Sequential()
        dropout = 0.4
        depth = 512
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.generator.add(Dense(dim*dim*depth, input_dim=100))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        self.generator.add(Reshape((dim, dim, depth)))
        self.generator.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.generator.add(Conv2DTranspose(1, 5, padding='same'))
        self.generator.add(Activation('sigmoid'))
        self.generator.summary()
        return self.generator

    def get_model(self):
        """
        The outer loop according to GTN paper
        :return: a compiled keras.Models.Sequential
        """
        if self.model:
            return self.model
        
        # ('outer loop' from the GTN paper)
        # real training data
        real_input = Input(shape=self.real_input_shape, name='real_input')
        real_learner = self.get_learner()(real_input)
        real_output = Dense(self.n_classes, activation='sigmoid', name='real_output')(real_learner)

        # ('inner loop' from the GTN paper)
        # noise input
        fake_input = Input(shape=self.noise_shape, name='fake_input'))
        teacher = self.get_generator()(fake_input)
        fake_learner = self.get_learner()(teacher)
        fake_output = Dense(self.n_classes, activation='sigmoid', name='fake_output')(teacher)

        self.model = Model(
            inputs=[real_input, fake_input],
            outputs=[real_output, fake_output]
        )

        # TODO: make loss/weights an instance variable?
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'real_output': 'categorical_crossentropy',
                'fake_output': 'categorical_crossentropy'
            },
            loss_weights=[
                'real_output': 1.0,
                'fake_output': 0.01
            ]
        )
        return self.model

    def train(self, fake_epochs, real_epochs):
        pass
