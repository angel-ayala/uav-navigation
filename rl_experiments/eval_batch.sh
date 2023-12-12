#!/bin/bash
python ./eval.py \
--seed 30 \
--logs-path './drone_vector/logs/ddqn-srl_2023-11-28_00-13-33' \
--episode 1

python ./eval.py \
--seed 30 \
--logs-path './drone_vector/logs/ddqn-srl_2023-11-28_00-13-33' \
--episode 2
