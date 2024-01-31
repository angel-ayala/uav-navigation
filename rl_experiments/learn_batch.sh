#!/bin/bash
# DDQN vector-based (13, )
python learn.py --memory-prioritized
# DDQN image-based (3, 84, 84)
python learn.py --memory-prioritized --is-pixels --use-cuda
# DDQN-SRL vector-based (13, )
python learn.py --memory-prioritized --is-srl
# DDQN-SRL image-based (3, 84, 84)
python learn.py --memory-prioritized --is-srl --is-pixels --use-cuda

# DDQN DEBUG
python learn.py --steps 1000 --eval-interval 100 --epsilon-steps 500 --memory-steps 64 --target-update-frequency 10 \
--is-pixels --is-srl --use-cuda
