import os
import numpy as np
import time
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate as ConcatLayer
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow import GradientTape

class GTN(object):
    def __init__(self, datagen=None,real_input_shape=None, n_classes=None, noise_shape=None, optimizer=None, save_synthetic=None):
        if datagen is None and real_input_shape is None and n_classes is None:
            raise ValueError("GTN requires Keras data generator (or keras.utils.Sequence) OR real_input_shape and n_classes")

        self.datagen = datagen

        # shape of the real data without batch size
        self.real_input_shape = real_input_shape

        # number of classes
        self.n_classes = datagen.n_classes if datagen else n_classes

        # shape of the noise vector for the generator
        if datagen.noise_shape:  # datagenerator noise shape has highest priority
            self.noise_shape = datagen.noise_shape
        elif noise_shape:  # provided noise shape is next priority
            self.noise_shape = noise_shape
        else:  # default
            self.noise_shape = (100,)
        
        # optimizer for both inner and outer loops...
        self.optimizer = RMSprop() if optimizer is None else optimizer

        # populated with functions below
        self.learner = None
        self.generator = None
        self.model = None

        self.synthetic_dir = save_synthetic  # get_or_create_dir(sythentic_dir) if sythentic_dir is not None else None

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
        noise_input = Input(
            shape=self.noise_shape,
            name='noise_input'
        )
        
        # (for 'outer loop' from the GTN paper)
        # real training data
        real_input = Input(
            shape=self.real_input_shape,
            name='real_input'
        )
        # generate takes noise as input
        teacher = self.get_generator(noise_input)

        concat_layer = ConcatLayer([teacher, real_input])
        gate_layer = Lambda(lambda x: x[:, :, :, 0:1], name="data_gate")(concat_layer)

        # learner gets input from synthetic data or real data
        learner = self.get_learner()

        x = learner(gate_layer)
        
        # learner should have a fake output so we can prevent fake data from updating weights
        output = Dense(self.n_classes, activation='sigmoid', name='output')(x)

        # model is compiled with two inputs and two outputs
        self.model = Model(
            inputs=[real_input, noise_input],
            outputs=[output, teacher]
        )

        return self.model

    def get_noise_array(self, batch_size):
        """
        TODO: should be used by manual training, not yet supported
        Make noise data for the generator
        :return: an np.array of shape self.noise_shape
        """
        n = 1
        for dimension in self.noise_shape:
            n = n * dimension
        return np.random.uniform(-1.0, 1.0, size=(batch_size,)+self.noise_shape)

    def train(self, inner_loops, outer_loops):
        model = self.get_model()

        optimizer = SGD()
        loss_func = CategoricalCrossentropy()
        
        # TODO: metrics

        learner = self.get_learner()

        epochs = 2
        for _ in range(outer_loops):
            for __ in range(inner_loops):
                for step, data in enumerate(self.datagen):
                    if step == 2: break
                    with GradientTape() as tape:
                        x, y = data
                        
                        # logits is shape (2, batch_size, num_classes)
                        # 2 is because of two inputs
                        logits = model(x)
                        loss_value = loss_func(y['output'], logits[0])
                    
                    
                    gradients = tape.gradient(loss_value, learner.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, learner.trainable_weights))
                    
                    
                    




    # TODO: support manually training
    def manual_train(self, fake_epochs, real_epochs, generator_batch_size=32):
        raise NotImplementedError
        
        model = self.get_model()

        inner_loops = 8
        # inner loop (fake data)
        for _ in range(inner_loops):
            # teacher data
            noise = self.get_noise_array(generator_batch_size)
            fake_classes = np.random.randint(self.n_classes, size=generator_batch_size)
            fake_classes = to_categorical(fake_classes)

            # blank data for the real input
            blank_input = np.zeros((generator_batch_size,)+self.real_input_shape)
            blank_output = np.zeros(generator_batch_size)
            blank_output = to_categorical(blank_output, num_classes=self.n_classes)
            
            model.fit(
                {'noise_input': noise, 'real_input': blank_input},
                {'fake_output': fake_classes, 'real_output': blank_output},
                epochs=2,
                batch_size=generator_batch_size
            )
