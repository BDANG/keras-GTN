# keras-GTN
An example implementation of [Uber's Generative Teaching Network (GTN)](https://eng.uber.com/generative-teaching-networks/) with [Keras (tensorflow)](https://keras.io)

**Currently have not tried exact reproduction**


### Setup (currently tested on Ubuntu w/ Tensorflow Docker)

* GPU:

`docker build -f gpuDockerfile -t kerasgtn .`

* CPU:

`docker build -f cpuDockerfile -t kerasgtn .`

### Run (bash)

`docker run -u $(id -u):$(id -g) -it --rm -v $PWD:/tf kerasgtn:latest bash`

Include `--gpus all` if using GPU Dockerfile

### Run (jupyter notebook)

`docker run -p 8888:8888 -u $(id -u):$(id -g) -it --rm -v $PWD:/tf kerasgtn:latest`

Include `--gpus all` if using GPU Dockerfile
