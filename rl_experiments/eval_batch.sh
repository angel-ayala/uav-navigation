#!/bin/bash
python eval.py --logspath './logs_drone_vector/ddqn-srl_2023-12-12_19-36-38' --load-config
python eval_td3_cf.py --logspath 'logs_cf_vector/td3-srl_2024-11-19_17-33-24' --load-config
