FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN python -m pip install --upgrade pip

RUN pip install advertorch
RUN pip install --upgrade git+https://github.com/BorealisAI/advertorch.git
RUN pip install einops
RUN pip install envpool
RUN pip install gym
RUN pip install matplotlib
RUN pip install mediapy
RUN pip install numpy
# RUN pip install opencv-python # TODO: had issues when installing opencv
RUN pip install tensorboard
RUN pip install tyro
