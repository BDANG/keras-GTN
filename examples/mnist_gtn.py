from kerasgtn.gtn import GTN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate as ConcatLayer
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import random


class MNISTDataGenerator(Sequence):
    def __init__(self, batch_size=16, n_classes=10, noise_shape=(100,), shuffle=False):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.noise_shape = noise_shape
        self.shuffle = shuffle
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # use small datset for the sake of development
        x_train = x_train[:1000, :, :]

        # channel last reshape
        x_train = x_train.reshape((1000, 28, 28, 1))
        y_train = y_train[:1000]

        self.x_train = x_train
        self.y_train = to_categorical(y_train, num_classes=self.n_classes)
        
        # toggle whether datagenerator should return noise only
        self.noise_only = True
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.x_train.shape[0] // self.batch_size)

    def __getitem__(self, index):
        """
        Generate one batch of data
        :return: {'real_input': real_data, 'noise_input': noise},
                 {'real_output': real_output, 'fake_output': fake_output}
        """
        if self.noise_only:
            # blank data
            real_data = np.zeros((self.batch_size,)+self.x_train.shape[1:])
            real_output = np.zeros(self.batch_size)
            real_output = to_categorical(real_output, num_classes=self.n_classes)
        else:
            real_data = self.x_train[index:(index+self.batch_size), :, :]
            real_output = self.y_train[index:(index+self.batch_size), :]
        
        noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, 100))
        fake_output = np.random.randint(self.n_classes, size=self.batch_size)
        fake_output = to_categorical(fake_output, num_classes=self.n_classes)
        
        return {'real_input': real_data, 'noise_input': noise}, {'real_output': real_output, 'fake_output': fake_output}
    
    def on_epoch_end(self):
        pass


class MNIST_GTN(GTN):
    def __init__(self, **kwargs):
        super(MNIST_GTN, self).__init__(**kwargs)
    
    def get_generator(self, input_layer):
        if self.generator is not None:
            return self.generator
        dropout = 0.4
        depth = 256
        dim = 7
        # In: (100,) 
        # Out: dim x dim x depth
        x = Dense(dim*dim*depth)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Reshape((dim, dim, depth))(x)
        x = Dropout(dropout)(x)

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        x = UpSampling2D()(x)
        x = Conv2DTranspose(int(depth/2), 5, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(int(depth/4), 5, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(int(depth/8), 5, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        x = Conv2DTranspose(1, 5, padding='same')(x)
        x = Activation('sigmoid')(x)
        self.generator = x
        return self.generator

    def get_learner(self, real_input, teacher):
        if self.learner is not None:
            return self.learner

        # learner has 2 possible inputs: real data and synthetic data (output of the generator)
        # TODO: verify concatenate axis
        x = ConcatLayer([real_input, teacher], axis=-1)
        
        # TODO: verify that this architecture is good for MNIST
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        x = Flatten()(x)

        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        self.learner = x
        return self.learner


if __name__ == "__main__":
    datagen = MNISTDataGenerator(n_classes=10)
    gtn = MNIST_GTN(datagen=datagen, real_input_shape=(28, 28, 1), n_classes=10)
    model = gtn.get_model()
    model.summary()
    gtn.train(inner_loops=4, outer_loops=2)
    