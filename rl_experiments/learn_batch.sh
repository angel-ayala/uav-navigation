#!/bin/bash
# DDQN DEBUG
python learn.py \
--frame-stack 3 \
--zone-steps 10 \
--is-vector \
--add-target-pos \
--uav-data=imu,gyro,gps,gps_vel,north \
--epsilon-steps 1000 \
--target-update-frequency 30 \
--memory-prioritized \
--is-srl \
--model-vector \
--steps 3000 \
--eval-interval 300 \
--eval-steps 0 \
--use-cuda \

# DDQN vector-based (16,)
python learn.py \
--frame-stack 3 \
--zone-steps 10 \
--is-vector \
--add-target-pos \
--uav-data=imu,gyro,gps,gps_vel,north \
--target-update-frequency 300 \
--memory-prioritized \
--eval-steps 0 \
--use-cuda \
--seed 202401

# DDQN-SRL vector-based (16,)
python learn.py \
--frame-stack 3 \
--zone-steps 10 \
--is-vector \
--add-target-pos \
--uav-data=imu,gyro,gps,gps_vel,north \
--target-update-frequency 300 \
--memory-prioritized \
--is-srl \
--model-vector \
--eval-steps 0 \
--use-cuda \
--seed 202401

# TD3 vector-based (16,)
python learn_td3_cf.py \
--frame-stack 3 \
--zone-steps 10 \
--is-vector \                        
--add-target-pos \
--uav-data=imu,gyro,gps,gps_vel,north \
--memory-steps 128 \
--eval-steps 0 \
--use-cuda \
--seed 202401

# TD3-SRL-Dist vector-based (16,) can be used with SAC also
python learn_td3_cf.py \
--frame-stack 3 \
--zone-steps 10 \
--is-vector \                        
--add-target-pos \
--uav-data=imu,gyro,gps,gps_vel,north \
--is-srl --model-vector-target-dist \
--eval-steps 0 \
--use-cuda \
--seed 202401

