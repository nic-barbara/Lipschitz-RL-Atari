#!/bin/bash

networks=("cnn" "lbdn" "orthogonal" "aol" "spectral")
lipschitz=(10.0 20.0 60.0 100.0)
seeds=(1 2 3 4)

for s in "${seeds[@]}"; do
    for network in "${networks[@]}"; do
        for g in "${lipschitz[@]}"; do
            python liprl/ppo_atari_envpool.py \
            --env-id "Pong-v5" \
            --network $network \
            --seed $s \
            --lipschitz $g \
            --save_dir "results/params/"
            if [ "$network" == "cnn" ]; then
                break
            fi
        done
    done
done
