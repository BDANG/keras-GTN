FROM tensorflow/tensorflow:latest-py3-jupyter

COPY setup.py /tf/setup.py

COPY kerasgtn /tf/kerasgtn

WORKDIR /tf

RUN pip install -e .
