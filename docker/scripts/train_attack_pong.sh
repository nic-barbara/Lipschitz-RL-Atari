#!/bin/bash

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/project/root/directory:/code lipschitz_rl_atari:latest \
    /bin/sh -c "
    pwd; 
    cd /code/;
    pip install -e .;
    ./scripts/s0_train_all.sh;
    python scripts/s1_run_attacks.py
    "
