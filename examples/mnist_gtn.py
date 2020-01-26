from kerasgtn.gtn import GTN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate as ConcatLayer
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop

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
        # TODO: verify concatenate axis
        x = ConcatLayer([real_input, teacher], axis=-1)
        
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
    gtn = MNIST_GTN(real_input_shape=(28, 28, 1), n_classes=10)
    model = gtn.get_model()
    model.summary()
    gtn.train(0, 0)
    