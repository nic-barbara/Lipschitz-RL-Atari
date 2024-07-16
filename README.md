# Lipschitz-Bounded Policy Networks for RL: Atari Pong

This repository contains the code used to produce the results in Section 4.2 of the [paper](https://arxiv.org/abs/2405.11432): *On Robust Reinforcement Learning with Lipschitz-Bounded Policy Networks*. See [here](https://github.com/nic-barbara/Lipschitz-RL-MJX) for the code used to produce the results in Section 4.1.

The code in this repository has been structured so that it is extensible to training and evaluating Lipschitz-bounded policies on Atari environments other than Pong. Please feel free to install it and play around with your favourite Atari games or re-create the figures from our paper.

## Quick results

We compare Lipschitz-bounded policy networks with standard, unconstrained CNNs on Atari Pong. Using a Lipschitz-bounded network makes the policy significantly more robust to noise and adversarial attacks. Here's an example of a CNN losing the game with a small amount of random noise in each image.

https://github.com/user-attachments/assets/f760c5d0-dcf6-46e4-811d-2af6dd6189b5

Much larger amounts of noise are required to make a Lipschitz-bounded (Sandwich) policy lose the game.

https://github.com/user-attachments/assets/c9d72812-d150-4f25-bc2e-0ed9dda82193

## Installation

This codebase is based on RL implementations in [`cleanrl`](https://github.com/vwxyzjn/cleanrl). There are two ways to install all the dependencies. We recommend the following as the easiest way to get started:
- Use a local install in a virtual environment for all development and results analysis.
- Use the docker image for training models on a server/cluster/different workstation GPU.

Note that the training code is currently not configured for multi-GPU training in PyTorch.

### System dependencies

All code was tested and developed in Ubuntu 22.04 with CUDA 12.3 and Python 3.10.12. To run any of the code in this repository, you will need a CUDA-compatible NVIDIA GPU.


### Using Python virtual environments

Create a Python virtual environment and activate it:

    python -m venv venv
    source venv/bin/activate

Install all dependencies and the project package itself (from the project root directory). Note that special installation is required for `advertorch`, whose base `pip` version seems to have some issues with later versions of PyTorch.

    pip install -r requirements.txt
    pip install advertorch
    pip install --upgrade git+https://github.com/BorealisAI/advertorch.git
    pip install -e .

Make sure all the directories have been created for saving results:

    ./scripts/make_directories.sh

Run training scripts as you see fit:

    ./scripts/train_all.sh

All evaluation plots use LaTeX font as the default. This requires a local install of LaTeX. Note that the default distribution of LaTeX on Ubuntu is not enough. Install the extra packages with the following:

    sudo apt update && sudo apt upgrade
    sudo apt install texlive-full

### Using Docker

Build the docker container from the project root directory (this might take a while):

    cd <repo/root/directory/>
    docker build -t lipschitz_rl_atari docker/

This will install all dependencies (including CUDA). Make sure all the directories have been created for saving results:

    ./scripts/make_directories.sh

Run a training script with docker:

    ./docker/scripts/train_attack_pong.sh

Note that you will first need to specify your local path to the repository in the training script.


## Repository structure

The repository is structured as follows:

- `docker/`: contains the Dockerfile to build the `docker` image, and also contains useful scripts with which to run training experiments from within the `docker` image.

- `liprl/`: contains all the tools used for to run experiments, collect data, and analyse results wrapped up into a (locally) pip-installable package.

- `results/`: contains select results files from my experiments. These include:
    - Trained models on each environment;
    - Adversarial attacks for each trained model on each environment;
    - Results plots; and
    - Plots used in the SysDO submission.

- `results-paper/`: contains a selection of trained models and results:
    - `results/attack-results/`: contains adversarial attack (and uniform random noise) results.
    - `results/params`: contains trained models.
    - `results/plots`: contains figures included in Section 4.2 of the paper.
    - `results/videos`: contains some gameplay videos with/without perturbations.

- `scripts/`: contains all the scripts used to train models, collect/aggregate data, and plot results. These are the runnable files which use the tools in the `liprl` package, and are numbered in the order they should be run to re-generate all the results from the paper.

Pre-trained models and attack results are only provided for a single model per architecture/Lipschitz bound rather than the 4 models for each used to produce the results in the paper. This is to limit the file size in this repository. Feel free to reach out to obtain the full set of trained models/attack results if you're interested!


## A few quick notes

### Naming conventions

Section 4.2 of the paper compares unconstrained CNN policies with various Lipschitz-bounded policies architectures. We use slightly different names in the code compared to the paper:
- Spectral Normalisation (SN) is referred to as `spectral`
- Almost Orthogonal Lipschitz (AOL) is referred to as `aol`
- Cayley layers are referred to as `orthogonal`
- Sandwich layers are referred to as `lbdn`

The term LBDN stands for Lipschitz-Bounded Deep Network (see [here](https://github.com/acfr/LBDN)).

### Model efficiency

It is **highly** recommended to put all the Lipschitz-bounded models into eval mode for RL rollouts, then training mode for batch updates (the current implementation of the code does this). Otherwise, the transformations due to the model parameterisations are performed unnecessarily during a rollout while the model parameters have not been not modified.


## Contact

Please contact Nicholas Barbara (nicholas.barbara@sydney.edu.au) with any questions.
