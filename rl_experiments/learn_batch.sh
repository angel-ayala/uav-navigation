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
--steps 3000 \
--eval-interval 300 \
--eval-steps 300 \
--epsilon-steps 1000 \
--target-update-frequency 300 \
--memory-capacity 2048 \
--memory-steps 32 \
--memory-prioritized \
--frame-stack 3 \
--is-pixels --use-cuda \
--is-srl
