# `VERY WIP` Currently have not tried exact reproduction

# keras-GTN
An example implementation of [Uber's Generative Teaching Network (GTN)](https://eng.uber.com/generative-teaching-networks/) with [Keras (tensorflow)](https://keras.io)

### Setup (currently tested on Ubuntu w/ Tensorflow Docker)

* GPU: `docker build -f gpuDockerfile -t kerasgtn .`

* CPU: `docker build -f cpuDockerfile -t kerasgtn .`

### Run Docker (bash)

`docker run --gpus all -u $(id -u):$(id -g) -it --rm -v $PWD:/tf kerasgtn:latest bash`

Remove `--gpus all` if using CPU Dockerfile

### Run Docker (jupyter notebook)

`docker run --gpus all -u $(id -u):$(id -g) -it --rm -v -p 8888:8888 $PWD:/tf kerasgtn:latest`

Remove `--gpus all` if using CPU Dockerfile

---

## *EXPECTED* Usage

```
from kerasgtn.gtn import GTN


class MyGTN(GTN):
    def __init__(self, **kwargs):
        super(MyGTN, self).__init__(**kwargs)
    
    def get_generator(self, input_layer):
        <implement>
    
    def get_learner(self, real_input, teacher):
        <implement>


gtn = MyGTN(input_shape=input_shape, n_classes=n_classes)
gtn.train(...)
```
