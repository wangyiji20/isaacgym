import numpy as np
import os
import random
import math
from isaacgym import gymtorch
from isaacgym.gymtorch import *
from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import *
from infrastructure.xenv import Xenv
import torch
import argparse
from rsl_rl.runners import OnPolicyRunner


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str)
parser.add_argument('--exp_name', type=str, default='todo')
parser.add_argument('--n_iter', '-n', type=int, default=3)      ############
parser.add_argument('--num_terrains',type=int,default=20)
parser.add_argument('--terrain_width',type=float,default=12.)     #integer
parser.add_argument('--horizontal_scale',type=float,default=0.25)  # [m]
parser.add_argument('--vertical_scale',type=float,default=0.005) # [m]
parser.add_argument('--num_envs',type=int)

args = parser.parse_args()
# convert to dictionary
params = vars(args)
params['num_envs']=2000
params['terrain_proportion']=[0.1,0.1,0.35,0.25,0.2]  

params['num_levels']=10
params['friction_range']=[0.2, 1.25]
params['Kp_range']=[50,60]
params['Kd_range']=[0.4,0.8]
params['motor_strength_range']=[0.9,1.1]
params['payload_range']=[0,6]
params['resample_prob']=0.004

train_cfg={}
train_cfg['policy']={}
train_cfg['policy']['init_noise_std'] = 1.0      #######
train_cfg['policy']['actor_hidden_dims']=[128, 128]
train_cfg['policy']['encoder_hidden_dims']=[256, 128]
train_cfg['policy']['critic_hidden_dims']=[256, 256, 256]
train_cfg['policy']['activation'] = 'elu'

train_cfg['algorithm']={}
train_cfg['algorithm']['value_loss_coef'] = 0.5
train_cfg['algorithm']['use_clipped_value_loss'] = True
train_cfg['algorithm']['clip_param'] = 0.2
train_cfg['algorithm']['entropy_coef'] = 0.
train_cfg['algorithm']['num_learning_epochs'] = 4
train_cfg['algorithm']['num_mini_batches']=4
train_cfg['algorithm']['learning_rate']=5.e-4
train_cfg['algorithm']['schedule'] = 'fixed'   ######
train_cfg['algorithm']['gamma'] = 0.998
train_cfg['algorithm']['lam'] = 0.95
train_cfg['algorithm']['desired_kl'] = 0.01
train_cfg['algorithm']['max_grad_norm'] = 1.

train_cfg['runner']={}
train_cfg['runner']['policy_class_name'] = 'ActorCritic'
train_cfg['runner']['algorithm_class_name'] = 'PPO'
train_cfg['runner']['num_steps_per_env'] = 20 # per iteration
train_cfg['runner']['max_iterations'] = 3
train_cfg['runner']['save_interval'] = 1000
train_cfg['runner']['experiment_name'] = 'test'
train_cfg['runner']['run_name'] = ''


xenv=Xenv(params)
trainer=OnPolicyRunner(xenv,train_cfg,log_dir='./runs')       
trainer.learn(train_cfg['runner']['max_iterations'])





   
