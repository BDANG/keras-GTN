from kerasgtn.gtn import GTN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate as ConcatLayer
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import random


class MNISTDataGenerator(Sequence):
    def __init__(self, batch_size=16, n_classes=10, noise_shape=(100,), shuffle=False):
        self.batch_size = batch_size
        
        # include an additional dead-class for when blank real data is given to the model
        self.n_classes = n_classes
        self.noise_shape = noise_shape
        self.shuffle = shuffle
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # use small datset for the sake of development
        sample_size = 1000
        x_train = x_train[:sample_size, :, :]

        # channel last reshape
        x_train = x_train.reshape((sample_size, 28, 28, 1))
        y_train = y_train[:sample_size]

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
        noise = np.random.normal(size=(self.batch_size, 100))

        
        if self.noise_only:
            # blank data
            real_data = np.zeros((self.batch_size,)+self.x_train.shape[1:])
            
            output = np.random.randint(self.n_classes, size=self.batch_size)
            output = to_categorical(output, num_classes=self.n_classes)            
        else:
            real_data = self.x_train[index:(index+self.batch_size), :, :]
            output = self.y_train[index:(index+self.batch_size), :]
        
        return {'real_input': real_data.astype(np.float32), 'noise_input': noise}, {'output': output}
    
    def on_epoch_end(self):
        pass


class MNISTLearner(Layer):
    def __init__(self):
        super(MNISTLearner, self).__init__()

        self.l1 = Conv2D(64, (3, 3), padding='same')
        self.l2 = LeakyReLU()
        self.l3 = MaxPooling2D(pool_size=(2,2))
        self.l4 = Flatten()
        self.l5 = Dense(128, activation='relu')
        self.l6 = Dropout(0.2)
        

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return self.l6(x)

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
        x = Activation('sigmoid', name="generator_output")(x)
        self.generator = x
        return self.generator

    def get_learner(self):
        if self.learner is not None:
            return self.learner
        
        # x = Conv2D(64, (3, 3), padding='same')(layer)
        # x = LeakyReLU()(x)
        # x = MaxPooling2D(pool_size=(2,2))(x)
        # x = Flatten()(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dropout(0.2)(x)
        # self.learner = x
        self.learner = MNISTLearner()
        return self.learner


if __name__ == "__main__":
    datagen = MNISTDataGenerator(n_classes=10)
    optimizer = Adadelta()
    gtn = MNIST_GTN(datagen=datagen, optimizer=optimizer, real_input_shape=(28, 28, 1), n_classes=10, save_synthetic="synthetic")
    model = gtn.get_model()
    model.summary()
    gtn.train(inner_loops=1, outer_loops=1)
    