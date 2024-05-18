#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

mkdir runs/
mkdir results/
mkdir results/attack-results/
mkdir results/params/
mkdir results/plots/
mkdir results/videos/