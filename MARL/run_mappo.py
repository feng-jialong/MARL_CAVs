#%%
'''
import sys
if "../" not in sys.path:
    sys.path.insert(0,"../")
if "./single_agent/" not in sys.path:
    sys.path.insert(0,"./single_agent/")
'''  
import importlib as imp

from torch.utils.tensorboard import SummaryWriter
from common.utils import agg_double_list, copy_file_ppo, init_dir
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser
import copy
import os
from datetime import datetime

from ..highway_env.envs import highway_env_v1
from .single_agent import Model_common
from .single_agent import Memory_common
import MaPPO
from MaPPO import MAPPO

# region 重新加载包和模块
imp.reload(gym.envs.registration)
imp.reload(highway_env_v1)  # 环境的重载还不行
imp.reload(MaPPO)
imp.reload(Memory_common)
imp.reload(Model_common)
imp.reload(highway_env.envs.common.observation)
imp.reload(highway_env.envs.common.abstract)
# endregion

def parse_args():
    """
    Description for this experiment:
    """
    # get the experiment name
    config_dir = 'configs/configs_ppo.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)
    name = config.get('OTHER_CONFIG','name')
    
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_ppo.ini'  # 配置文件的路径
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using mappo'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--experiment-dir', type=str, required=False,
                        default=default_base_dir+name+'/', help="experiment dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")  # train模式或evaluate模式
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default='./results/'+name+'/models/', help="pretrained model path") # 预训练模型的路径，用于断点续训
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    # args = parser.parse_args()
    args =parser.parse_known_args()[0]
    return args

def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder with time and date
    # now = datetime.utcnow().strftime(r"%b-%d_%H-%M-%S")
    # create an experiment folder with name
    name = config.get('OTHER_CONFIG','name')
    now = config.get('OTHER_CONFIG','name')
    logger = SummaryWriter('./results/'+name+'/logs/')
    
    output_dir = base_dir + now   # 输出路径，同时也是当次实验的路径
    dirs = init_dir(output_dir)
    copy_file_ppo(dirs['configs'])

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

    # region train configs
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    # endregion

    # region 配置train环境
    env = gym.make('highway-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getfloat('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getfloat('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['FLOCKING_REWARD'] = config.getfloat('ENV_CONFIG', 'FLOCKING_REWARD')
    env.config['DESIRE_SPEED'] = config.getfloat('ENV_CONFIG','DESIRE_SPEED')
    env.config['SAFE_DECISION_REWARD'] = config.getfloat('ENV_CONFIG', 'SAFE_DECISION_REWARD')
    env.config['lanes_number'] = config.getint('ENV_CONFIG', 'lanes_number')
    env.config['CAV_number'] = config.getint('ENV_CONFIG', 'CAV_number')
    env.config['COMFORT_REWARD'] = config.getfloat('ENV_CONFIG', 'COMFORT_REWARD')
    # endregion

    # region 配置evaluate环境
    env_eval = gym.make('highway-multi-agent-v0')  
    env_eval.config = copy.deepcopy(env.config)
    env_eval.use_centralized_critic = config.getboolean('MODEL_CONFIG','use_centralized_critic')
    '''
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env_eval.config['COLLISION_REWARD'] = config.getfloat('ENV_CONFIG', 'COLLISION_REWARD')
    env_eval.config['HIGH_SPEED_REWARD'] = config.getfloat('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env_eval.config['FLOCKING_REWARD'] = config.getfloat('ENV_CONFIG', 'FLOCKING_REWARD')
    env_eval.config['FLOCKING_UB'] = config.getfloat('ENV_CONFIG', 'FLOCKING_UB')        
    env_eval.config['FLOCKING_LB'] = config.getfloat('ENV_CONFIG', 'FLOCKING_LB')
    env_eval.config['HEADWAY_COST'] = config.getfloat('ENV_CONFIG', 'HEADWAY_COST')
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env_eval.config['LANE_CHANGE_COST'] = config.getfloat('ENV_CONFIG', 'LANE_CHANGE_COST')
    env_eval.config['lanes_number'] = config.getint('ENV_CONFIG', 'lanes_number')
    env_eval.config['CAV_number'] = config.getint('ENV_CONFIG', 'CAV_number')
    env_eval.config['HDV_number'] = config.getint('ENV_CONFIG', 'HDV_number')
    env_eval.config['action_masking'] = config.getboolean('ENV_CONFIG', 'action_masking')
    env_eval.config['max_reward_speed_range'] = config.getfloat('ENV_CONFIG','max_reward_speed_range')
    env_eval.config['min_reward_speed_range'] = config.getfloat('ENV_CONFIG','min_reward_speed_range')
    '''
    # endregion

    # region 载入模型参数
    mappo_config = {}
    mappo_config['use_orthogonal_init'] = config.getboolean('MODEL_CONFIG','use_orthogonal_init')
    mappo_config['use_layer_norm'] = config.getboolean('MODEL_CONFIG','use_layer_norm')
    mappo_config['use_observation_norm'] = config.getboolean('MODEL_CONFIG','use_observation_norm')
    mappo_config['use_gae'] = True
    mappo_config['gae_lambda'] = config.getfloat('MODEL_CONFIG','gae_lambda')
    mappo_config['eps'] = config.getfloat('MODEL_CONFIG','eps')
    mappo_config['use_advantage_norm'] = config.getboolean('MODEL_CONFIG','use_advantage_norm')
    mappo_config['critic_loss'] = config.get('MODEL_CONFIG','critic_loss')
    mappo_config['use_value_clip'] = config.getboolean('MODEL_CONFIG','use_value_clip')
    mappo_config['use_ppo_clip'] = config.getboolean('MODEL_CONFIG','use_ppo_clip')
    mappo_config['use_kl_penalty'] = config.getboolean('MODEL_CONFIG','use_kl_penalty')
    mappo_config['kl_beta'] = config.getfloat('MODEL_CONFIG','kl_beta')
    mappo_config['use_advantage_clip'] = config.getboolean('MODEL_CONFIG','use_advantage_clip')
    mappo_config['use_value_norm'] = config.getboolean('MODEL_CONFIG','use_value_norm')
    mappo_config['use_value_loss_clip'] = config.getboolean('MODEL_CONFIG','use_value_loss_clip')
    mappo_config['use_policy_loss_clip'] = config.getboolean('MODEL_CONFIG','use_policy_loss_clip')
    mappo_config['use_observation_clip'] = config.getboolean('MODEL_CONFIG','use_observation_clip')
    mappo_config['epoch_num'] = config.getint('MODEL_CONFIG','epoch_num')
    mappo_config['activation_type'] = config.get('MODEL_CONFIG','activation_type')    
    mappo_config['scene_frequency'] = config.getint('MODEL_CONFIG','scene_frequency')    
    mappo_config['use_entropy_loss'] = config.getboolean('MODEL_CONFIG','use_entropy_loss') 
    mappo_config['entropy_loss_beta'] = config.getfloat('MODEL_CONFIG','entropy_loss_beta') 
    mappo_config['critic_target_type'] = config.get('MODEL_CONFIG','critic_target_type') 
    mappo_config['use_param_share'] = config.getboolean('MODEL_CONFIG','use_param_share')
    mappo_config['obs_dim'] = config.getint('MODEL_CONFIG','obs_dim')    
    mappo_config['actor_shared_size'] = config.getint('MODEL_CONFIG','actor_shared_size')  
    mappo_config['critic_shared_size'] = config.getint('MODEL_CONFIG','critic_shared_size')
    mappo_config['memory_capacity'] = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    mappo_config['batch_size'] = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    mappo_config['reuse_times'] = config.getint('MODEL_CONFIG','reuse_times')
    mappo_config['actor_hidden_size'] = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    mappo_config['critic_hidden_size'] = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    mappo_config['actor_lr'] = config.getfloat('MODEL_CONFIG', 'actor_lr')
    mappo_config['critic_lr'] = config.getfloat('MODEL_CONFIG', 'critic_lr')
    mappo_config['reward_scale'] = config.getfloat('MODEL_CONFIG', 'reward_scale')
    mappo_config['target_update_steps'] = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    mappo_config['target_tau'] = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')
    mappo_config['reward_gamma'] = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    mappo_config['reward_type'] = config.get('MODEL_CONFIG', 'reward_type')
    mappo_config['max_grad_norm'] = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    mappo_config['clip_param'] = config.getfloat('MODEL_CONFIG','clip_param')
    mappo_config['use_cuda'] = config.getboolean('MODEL_CONFIG','use_cuda')
    mappo_config['roll_out_n_episodes'] = config.getint('MODEL_CONFIG','ROLL_OUT_N_EPISODES')
    mappo_config['optimizer_type'] = config.get('MODEL_CONFIG','OPTIMIZER_TYPE')
    mappo_config['torch_seed'] = config.getint('MODEL_CONFIG','torch_seed')
    mappo_config['use_adaptive_kl_penalty'] = config.getboolean('MODEL_CONFIG','use_adaptive_kl_penalty')
    mappo_config['use_centralized_critic'] = config.getboolean('MODEL_CONFIG','use_centralized_critic')
    mappo_config['use_rnn'] = config.getboolean('MODEL_CONFIG','use_rnn')
    mappo_config['collision_replay_times'] = config.getint('MODEL_CONFIG','collision_replay_times')    
    mappo_config['max_obs'] = config.getint('MODEL_CONFIG','max_obs') 
    # endregion
    
    # region 初始化MAPPO
    mappo = MAPPO(env=env,
                  state_dim=env.n_s, 
                  action_dim=env.n_a,
                  logger=logger,
                  test_seeds=args.evaluation_seeds,
                  config=mappo_config)
    # endregion

    # load the model if exist
    mappo.load(model_dir, train_mode=True)
    env.seed = env.config['seed']
    eval_rewards = []
    
    # region train process
    while mappo.n_episodes < MAX_EPISODES:
        mappo.interact()
        if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
            mappo.train()
        if mappo.episode_done and ((mappo.n_episodes+mappo.global_step) % EVAL_INTERVAL == 0):
            rewards, _, _, _, _ = mappo.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)   # evaluation环境通过类方法使用
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (mappo.global_step+mappo.n_episodes, rewards_mu))  # 已经用于训练的episodes数量
            eval_rewards.append(rewards_mu)
            # save the model
            mappo.save(dirs['models'], mappo.n_episodes+mappo.global_step)
    # endregion
    
    # save the model
    mappo.save(dirs['models'], mappo.n_episodes+mappo.global_step) # 即时保存
    
    with open(base_dir+name+r'/rewards_per16.txt','a') as f:
        for reward in eval_rewards:
            f.write(str(reward)+'\n')

    logger.close()

def evaluate(args):  # 参数的读取需要修改
    
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.experiment_dir + 'configs/configs_ppo.ini'  # 参数采用预训练模型文件夹中的
    config = configparser.ConfigParser()
    config.read(config_dir)

    name = config.get('OTHER_CONFIG','name')
    logger = SummaryWriter('./results/'+name+'/logs/')
    video_dir = args.experiment_dir + '/eval_videos'
    
    # train configs
    EVAL_MAX_EPISODES = config.getint('EVAL_CONFIG', 'EVAL_MAX_EPISODES')
    
    # region 配置eval环境
    env = gym.make('highway-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getfloat('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getfloat('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['FLOCKING_REWARD'] = config.getfloat('ENV_CONFIG', 'FLOCKING_REWARD')
    env.config['FLOCKING_UB'] = config.getfloat('ENV_CONFIG', 'FLOCKING_UB')
    env.config['FLOCKING_LB'] = config.getfloat('ENV_CONFIG', 'FLOCKING_LB')
    env.config['HEADWAY_COST'] = config.getfloat('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['LANE_CHANGE_COST'] = config.getfloat('ENV_CONFIG', 'LANE_CHANGE_COST')
    env.config['lanes_number'] = config.getint('ENV_CONFIG', 'lanes_number')
    env.config['CAV_number'] = config.getint('ENV_CONFIG', 'CAV_number')
    env.config['HDV_number'] = config.getint('ENV_CONFIG', 'HDV_number')
    env.config['action_masking'] = config.getboolean('ENV_CONFIG', 'action_masking')
    env.config['max_reward_speed_range'] = config.getfloat('ENV_CONFIG','max_reward_speed_range')
    env.config['min_reward_speed_range'] = config.getfloat('ENV_CONFIG','min_reward_speed_range')
    # endregion

    mappo = MAPPO(env=env,
                  state_dim=env.n_s, 
                  action_dim=env.n_a,
                  logger=logger,
                  test_seeds=args.evaluation_seeds,
                  config=None)

    # load the model if exist
    mappo.load(model_dir, train_mode=False)
    rewards,steps,avg_speeds,flocking_distances,(vehicle_speeds,vehicle_positions) = mappo.evaluation(env, video_dir, EVAL_MAX_EPISODES, is_train=False)

    # 保存速度和位置轨迹
    np.save(args.experiment_dir+'eval_logs/'+'speed_trajectories.npy',np.array(vehicle_speeds))
    np.save(args.experiment_dir+'eval_logs/'+'position_trajectories.npy',np.array(vehicle_positions))
    
    # reward,平均速度,集群距离分布的可视化
    np.save(args.experiment_dir+'eval_logs/'+'rewards.npy',np.array(rewards))
    np.save(args.experiment_dir+'eval_logs/'+'avg_speeds.npy',np.array(avg_speeds))
    np.save(args.experiment_dir+'eval_logs/'+'flocking_disntances.npy',np.array(flocking_distances))
    
    # 计算collision rate
    T = config.getfloat('ENV_CONFIG','duration')*config.getfloat('ENV_CONFIG','policy_frequency')
    print('Success rate:',str(sum([1 if i==T else 0 for i in steps])))

#%% 
import cProfile
p = cProfile.Profile()
p.enable()#开始采集

if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
        
p.disable()#结束采集
# p.print_stats(sort='tottime')#打印结果

# region profile导出为文本文件
import pstats
with open("profilingStats-cum.txt","w") as f:
    ps = pstats.Stats(p, stream=f)
    ps.sort_stats('cumulative')
    ps.print_stats()
with open("profilingStats-tot.txt","w") as f:
    ps = pstats.Stats(p, stream=f)
    ps.sort_stats('tottime')
    ps.print_stats()
# endregion
#%%
'''
if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
'''
#%%
'''
with open(r'./results/'+'v3'+'/rewards_per100.txt','r') as f:
    rewards = [float(r) for r in f.readlines()]
plt.figure()
plt.plot(100*np.arange(len(rewards)),rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend(["MAPPO"])
'''

#%%
import torch
def batch_gather_vec(tensor, indices):
    shape = list(tensor.shape)
    flat_first = torch.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    offset = torch.reshape(torch.arange(shape[0]).cuda() * shape[1],[shape[0]] + [1] * (len(indices.shape) - 1))
    output = flat_first[indices + offset]
    return output