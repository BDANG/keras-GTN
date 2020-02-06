from tensorflow.keras.callbacks import Callback
import numpy as np

class SaveSyntheticCallback(Callback):
    def __init__(self, **kwargs):
        self.datagen = kwargs.pop('datagen')
        self.synthetic_dir = kwargs.pop('synthetic_dir')
        self.save_interval = kwargs.pop('save_interval')

        super(SaveSyntheticCallback, self).__init__(**kwargs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        if not self.synthetic_dir or not self.save_interval:
            return
        
        if batch % self.save_interval == 0:
            # send noise through 
            #self.model.summary()
            print(type(self.datagen[0][0]))
            #r = self.model.predict(self.datagen[0][0])


class PrintGeneratorWeightsCallback(Callback):
    def __init__(self, **kwargs):
        super(PrintGeneratorWeightsCallback, self).__init__(**kwargs)
    
    def on_train_batch_end(self, batch, logs=None):
        w = self.model.layers[1].get_weights()
        #print(self.model.layers[1].get_config())
        print(len(w[0]))
        print(np.array(w).shape)


