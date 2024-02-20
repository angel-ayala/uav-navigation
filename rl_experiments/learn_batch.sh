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
python learn.py \
--steps 1000 \
--eval-interval 100 \
--epsilon-steps 500 \
--target-update-frequency 10 \
--memory-capacity 128 \
--memory-steps 32 \
--memory-prioritized \
--target-pos random \
--frame-stack 3 \
--is-pixels --use-cuda \
--is-srl
