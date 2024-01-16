#!/bin/bash
# pixel-based DDQN
python learn.py --is-pixels

# pixel-based DDQN DEBUG
python learn.py --is-pixels --steps 1000 --eval-interval 100 --epsilon-decay-steps 500 --memory-steps 64 --target-update-frequency 10 --use-cuda

# pixel-based DDQN-SRL
python learn.py --is-pixels --is-srl --use-cuda

# vector-based DDQN
python learn.py

# vector-based DDQN-SRL
python learn.py --is-srl --hidden-dim 64 --num-layers 1
