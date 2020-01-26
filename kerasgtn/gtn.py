import numpy as np
import time


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate as ConcatLayer
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop


class GTN(object):
    def __init__(self, real_input_shape, n_classes, noise_shape=None, optimizer=None):

        # shape of the real data without batch size
        self.real_input_shape = real_input_shape

        # number of classes
        self.n_classes = n_classes

        # shape of the noise vector for the generator
        self.noise_shape = noise_shape if noise_shape else (100,)
        
        # optimizer for both inner and outer loops...
        self.optimizer = RMSprop() if optimizer is None else optimizer

        # populated with functions below
        self.learner = None
        self.generator = None
        self.model = None

    def get_learner(self, teacher, real_input):
        """
        :return: an uncompiled keras.models.Model that represents a learner
        """
        if self.learner:
            return self.learner
        raise NotImplementedError

    def get_generator(self, input_layer):
        """
        :return: an uncompiled keras.models.Model model that represents a generator
        """
        if self.generator:
            return self.generator
        raise NotImplementedError

    def get_model(self):
        """
        The outer loop according to GTN paper
        :return: a compiled keras.Models.Sequential
        """
        if self.model:
            return self.model
        
        # ('inner loop' from the GTN paper)
        # noise input
        fake_input = Input(shape=self.noise_shape, name='fake_input')
        
        # ('outer loop' from the GTN paper)
        # real training data
        real_input = Input(shape=self.real_input_shape, name='real_input')        
        
        teacher = self.get_generator(fake_input)
        learner = self.get_learner(teacher, real_input)
        
        fake_output = Dense(self.n_classes, activation='sigmoid', name='fake_output')(learner)
        real_output = Dense(self.n_classes, activation='sigmoid', name='real_output')(learner)

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
            loss_weights={
                'real_output': 1.0,
                'fake_output': 0.01
            }
        )
        return self.model

    def train(self, fake_epochs, real_epochs):
        pass
