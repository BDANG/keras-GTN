import numpy as np
import time


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate as ConcatLayer
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from tensorflow.keras.utils import to_categorical

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
        The full GTN architecture
        :return: a compiled keras.model.Model
        """
        if self.model:
            return self.model
        
        # (for the 'inner loop' from the GTN paper)
        # noise input
        noise_input = Input(shape=self.noise_shape, name='noise_input')
        
        # (for 'outer loop' from the GTN paper)
        # real training data
        real_input = Input(shape=self.real_input_shape, name='real_input')        
        
        # generate takes noise as input
        teacher = self.get_generator(noise_input)

        # learner gets input from synthetic data or real data
        learner = self.get_learner(teacher, real_input)
        
        # learner should have a fake output so we can prevent fake data from updating weights
        fake_output = Dense(self.n_classes, activation='sigmoid', name='fake_output')(learner)
        real_output = Dense(self.n_classes, activation='sigmoid', name='real_output')(learner)

        self.model = Model(
            inputs=[real_input, noise_input],
            outputs=[real_output, fake_output]
        )

        # TODO: make loss/weights a class variable?
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'real_output': 'categorical_crossentropy',
                'fake_output': 'categorical_crossentropy'
            },
            loss_weights={
                'real_output': 1.0,
                'fake_output': 1.0
            }
        )
        return self.model

    def get_noise_array(self, batch_size):
        """
        Make noise data for the generator
        :return: an np.array of shape self.noise_shape
        """
        n = 1
        for dimension in self.noise_shape:
            n = n * dimension
        return np.random.uniform(-1.0, 1.0, size=(batch_size,)+self.noise_shape)

    def train(self, fake_epochs, real_epochs, generator_batch_size=32):
        
        model = self.get_model()

        inner_loops = 8
        # inner loop (fake data)
        for _ in range(inner_loops):
            noise = self.get_noise_array(generator_batch_size)
            fake_classes = np.random.randint(self.n_classes, size=generator_batch_size)
            fake_classes = to_categorical(fake_classes)

            blank_input = np.zeros((generator_batch_size,)+self.real_input_shape)
            blank_output = np.zeros(generator_batch_size)
            blank_output = to_categorical(blank_output, num_classes=self.n_classes)
            
            model.fit(
                {'noise_input': noise, 'real_input': blank_input},
                {'fake_output': fake_classes, 'real_output': blank_output},
                epochs=2,
                batch_size=generator_batch_size
            )
