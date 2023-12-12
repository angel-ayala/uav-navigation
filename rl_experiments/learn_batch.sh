#!/bin/bash
# pixel-based DDQN
python learn.py --is-pixels

# pixel-based DDQN-SRL
python learn.py --is-pixels --is-srl

# vector-based DDQN
python learn.py

# vector-based DDQN-SRL
python learn.py --is-srl --hidden-dim 64 --num-layers 1