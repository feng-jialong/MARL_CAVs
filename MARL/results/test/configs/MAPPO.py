import torch as th
from torch import nn
import configparser

config_dir = 'configs/configs_ppo.ini'
config = configparser.ConfigParser()
config.read(config_dir)

torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

from torch.optim import Adam, RMSprop

import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Memory_common import OnPolicyReplayMemory
from single_agent.Model_common import ActorNetwork, CriticNetwork, ActorNetworkPlus, CriticNetworkPlus
from common.utils import index_to_one_hot, to_tensor_var, VideoRecorder


class MAPPO: # 默认继承object类
    """
    An multi-agent learned with PPO
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """
    def __init__(self, env, state_dim, action_dim,
                 logger=None,test_seeds=0,config={}):  # 可以直接传入配置字典

        self.config = config  # 传入配置更方便
        self.reward_type = self.config['reward_type']
        assert self.reward_type in ["regional_R", "global_R"]
        self.env = env   # 使用train的环境初始化MAPPO
        self.env_state, self.action_mask = self.env.reset()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_step = 0   # 用于记录断点续训，总体训练进度
        self.n_episodes = 0   # 用于记录每次训练的进度
        self.n_steps = 0   # 当前episode的决策步步数
        self.n_agents = len(self.env.controlled_vehicles)
        self.test_seeds = test_seeds
        self.reward_gamma = self.config['reward_gamma']
        self.reward_scale = self.config['reward_scale']
        self.memory = OnPolicyReplayMemory(self.config['memory_capacity'],self.reward_gamma)
        self.actor_hidden_size = self.config['actor_hidden_size']
        self.critic_hidden_size = self.config['critic_hidden_size']
        self.critic_loss = self.config['critic_loss']
        self.actor_lr = self.config['actor_lr']
        self.critic_lr = self.config['critic_lr']
        self.optimizer_type = self.config['optimizer_type']
        self.max_grad_norm = self.config['max_grad_norm']
        self.batch_size = self.config['batch_size']
        self.use_cuda = self.config['use_cuda'] and th.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.target_tau = self.config['target_tau']
        self.target_update_steps = self.config['target_update_steps']
        self.clip_param = self.config['clip_param']
        self.reuse_times = self.config['reuse_times']
        self.roll_out_n_episodes = self.config['roll_out_n_episodes']
        self.epoch_num = self.config['epoch_num']
        self.logger = logger
        
        if self.config['use_param_share']:
            self.actor = ActorNetworkPlus(self.config['obs_dim'],self.config['actor_shared_size'],
                                          self.actor_hidden_size,self.action_dim,self.config)
            self.critic = CriticNetworkPlus(self.config['obs_dim'],self.config['critic_shared_size'],
                                            self.critic_hidden_size,self.config)
        else:
            self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                    self.action_dim,self.config)
            self.critic = CriticNetwork(self.state_dim, self.critic_hidden_size,1,self.config)
            
        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(),lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(),lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(),lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(),lr=self.critic_lr)
        else:
            Exception('Invalid optimizer_type')

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.episode_rewards = [0]  # 每一episode的reward
        self.average_speed = [0]   # 
        self.episode_steps = [0]   # 每一episode的policy步数

    def interact(self):
        self.env.scene_type = int((self.n_episodes+self.global_step)/self.config['scene_frequency'])%12
        self.env.reset()  # 更新scene_type后强行reset一次
        
        for _ in range(self.roll_out_n_episodes):
            done = True
            average_speed = 0
            flocking_distance = 0
            
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            while(True):
                states.append(self.env_state)
                action = self.exploration_action(self.env_state, self.n_agents)
                next_state, global_reward, done, info = self.env.step(tuple(action))
                next_states.append(next_state)
                dones.append(done)
                actions.append([index_to_one_hot(a,self.action_dim) for a in action])
                self.episode_rewards[-1] += global_reward
                self.episode_steps[-1] += 1
                if self.reward_type == "regional_R":
                    reward = info["regional_rewards"]
                elif self.reward_type == "global_R":
                    reward = [global_reward] * self.n_agents
                rewards.append(reward)
                average_speed += info["average_speed"]  # 每一step的车均速度
                flocking_distance += info['flocking_distance']
                self.env_state = next_state
                self.n_steps += 1
                
                if done:
                    self.env_state, _ = self.env.reset()
                    self.n_episodes += 1
                    self.episode_done = True
                    self.n_steps = 0
                    self.average_speed[-1] = average_speed / self.episode_steps[-1]   # 整个episode的平均车速
                    flocking_distance /= self.episode_steps[-1]  # 整个episode的平均聚群距离
                    
                    # tesorboard logger: info of episodes
                    self.logger.add_scalar('info/reward',self.episode_rewards[-1],self.n_episodes+self.global_step)
                    self.logger.add_scalar('info/average_speed',self.average_speed[-1],self.n_episodes+self.global_step)
                    self.logger.add_scalar('info/flocking_distance',flocking_distance,self.n_episodes+self.global_step)
                    self.logger.add_scalar('info/episode_step',self.episode_steps[-1],self.n_episodes+self.global_step)
                    
                    self.episode_rewards.append(0)
                    self.average_speed.append(0)
                    self.episode_steps.append(0)
                    
                    # update actor target network and critic target network
                    # 在此处更新target network,为了能紧密跟随global episode的推进而更新
                    if (self.n_episodes+self.global_step) % self.target_update_steps == 0:
                        self._soft_update_target(self.actor_target, self.actor)
                        self._soft_update_target(self.critic_target, self.critic)
                    break

            # reward scaling
            if self.reward_scale is not None and self.reward_scale > 0:
                rewards = np.array(rewards) / self.reward_scale

            rewards = rewards.tolist()
            self.memory.push(states,actions,rewards,next_states,dones)
    
    def train(self):
        batch_size = min(len(self.memory),self.batch_size)  # the true size. self.batch_size is just a param
        batch = self.memory.sample(batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim).detach()
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim).detach()
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents,1).detach()
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim).detach()
        masks_var = to_tensor_var(batch.masks, self.use_cuda).view(-1).detach()
        # 本身及其子图不含参数的张量全部detach
        # 用于计算的初始数据全部detach
        
        # observation normalization 
        if self.config['use_observation_norm']:
            states_mean_var = states_var.mean(dim=0,keepdim=True)   # 默认在gpu中
            states_std_var = states_var.std(dim=0,keepdim=True)   # 默认在gpu中
            states_norm_var = ((states_var-states_mean_var)/(states_std_var+self.config['eps'])).detach()
            
            next_states_mean_var = next_states_var.mean(dim=0,keepdim=True)
            next_states_std_var = next_states_var.std(dim=0,keepdim=True)           
            next_states_norm_var = ((next_states_var-next_states_mean_var)/(next_states_std_var+self.config['eps'])).detach()
        else:
            states_norm_var = states_var
            next_states_norm_var = next_states_var
        # observation clip
        if self.config['use_observation_clip']:
            pass
        # GAE
        if self.config['use_gae']:  # always use gae and vary gae_lambda to switch to TD or Monte Carlo
            deltas_var = th.cuda.FloatTensor(batch_size,self.n_agents,1,device=self.device).detach()
            advantages_var = th.cuda.FloatTensor(batch_size,self.n_agents,1,device=self.device).detach()  # 分离出计算图
            returns_var = th.cuda.FloatTensor(batch_size,self.n_agents,1,device=self.device).detach()
            
            for agent_id in range(self.n_agents):
                prev_return = 0
                prev_advantage = 0
                # 递归求解
                for i in reversed(range(batch_size)):
                    returns_var[i,agent_id,:] = rewards_var[i,agent_id,:] + self.reward_gamma * prev_return * masks_var[i]
                    deltas_var[i,agent_id,:] = rewards_var[i,agent_id,:] \
                        + self.critic_target(next_states_norm_var[i,agent_id,:]) * masks_var[i] \
                            - self.critic_target(states_norm_var[i,agent_id,:])
                    advantages_var[i,agent_id,:] = deltas_var[i,agent_id,:] + self.config['gae_lambda'] * prev_advantage * masks_var[i]
                    prev_advantage = advantages_var[i,agent_id,:]
                    prev_return = returns_var[i,agent_id,:]
        advantages_var = advantages_var.detach()
        # advantage norm
        if self.config['use_advantage_norm']:
            advantages_var = ((advantages_var-advantages_var.mean(dim=0,keepdim=True))/(advantages_var.std(dim=0,keepdim=True)+self.config['eps'])).detach()
        # advantage clipp
        if self.config['use_advantage_clip']:
            pass
        
        for i in range(self.epoch_num):
            for agent_id in range(self.n_agents):
                ## update actor network
                self.actor_optimizer.zero_grad()
                action_log_pdf = self.actor(states_var[:, agent_id, :])  # do not normalize policy's inputs
                action_log_probs = th.sum(action_log_pdf * actions_var[:, agent_id,:],dim=-1)
                old_action_log_pdf = self.actor_target(states_var[:, agent_id,:])
                old_action_log_probs = th.sum(old_action_log_pdf * actions_var[:, agent_id,:],dim=-1).detach()
                ratio = th.exp(action_log_probs - old_action_log_probs)
                actor_loss = ratio * advantages_var[:,agent_id,0]
                # ppo clip
                if self.config['use_ppo_clip']:
                    clipped_loss = th.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * advantages_var[:,agent_id,0]
                    actor_loss = th.min(actor_loss,clipped_loss)
                actor_loss = -th.mean(actor_loss)
                # kl penalty
                if self.config['use_kl_penalty']:
                    kl_loss = nn.functional.kl_div(action_log_pdf,old_action_log_pdf,
                                                   reduction='batchmean',log_target=True)
                    # tensorboard logger: kl_loss
                    self.logger.add_scalars('loss/kl_loss',{str(agent_id):kl_loss},self.global_step+self.n_episodes)
                    actor_loss = actor_loss + self.config['kl_beta']*kl_loss
                # entropy loss
                if self.config['use_entropy_loss']:
                    entropy_loss = -1*th.mean(th.sum(th.exp(action_log_pdf)*action_log_pdf,dim=1))
                    self.logger.add_scalars('loss/entropy_loss',{str(agent_id):entropy_loss},self.global_step+self.n_episodes)
                    actor_loss = actor_loss - self.config['entropy_loss_beta']*entropy_loss
                # policy loss clip
                if self.config['use_policy_loss_clip']:
                    pass
                # tensorboard logger: actor loss
                self.logger.add_scalars('loss/actor_loss',{str(agent_id):actor_loss},self.global_step+self.n_episodes)
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                
                '''
                # tensorborad logger: actor parameters
                self.logger.add_histogram('actor/fc1/weight',self.actor.fc1.weight,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc1/bias',self.actor.fc1.bias,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc2/weight',self.actor.fc2.weight,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc2/bias',self.actor.fc2.bias,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc3/weight',self.actor.fc3.weight,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc3/bias',self.actor.fc3.bias,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc1/weight_grad',self.actor.fc1.weight.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc1/bias_grad',self.actor.fc1.bias.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc2/weight_grad',self.actor.fc2.weight.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc2/bias_grad',self.actor.fc2.bias.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc3/weight_grad',self.actor.fc3.weight.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('actor/fc3/bias_grad',self.actor.fc3.bias.grad,self.global_step+self.n_episodes)
                '''
                self.actor_optimizer.step()

                ## update critic network
                self.critic_optimizer.zero_grad()
                if self.config['critic_target_type'] == 'TD':
                    # TD方法模拟value function，需要critic target
                    next_values = self.critic_target(next_states_norm_var[:,agent_id,:])
                    target_values = rewards_var[:,agent_id,:] + self.reward_gamma * next_values  
                elif self.config['critic_target_type'] == 'MC':
                    # Monte Carlo方法模拟value function, 就不需要critic target了
                    target_values = returns_var[:,agent_id,:]
                target_values = target_values.detach()
                
                values_est = self.critic(states_norm_var[:,agent_id,:])  # critic估计value function
                    
                if self.config['critic_loss'] == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values_est, target_values)
                elif self.config['critic_loss'] == "MSE":
                    critic_loss = nn.MSELoss()(values_est, target_values)
                else:
                    raise ValueError('Invalid critic loss type')
                
                if self.config['use_value_clip']:  # 有问题,clamp函数的上下限必须是常数
                    critic_loss = th.min(critic_loss,nn.MSELoss(target_values,
                                                                th.clamp(values_est,
                                                                         values_est-self.clip_param,
                                                                         values_est+self.clip_param)))
                    
                if self.config['use_value_loss_clip']:
                    pass
                # tensorboard logger: critic loss
                self.logger.add_scalars('loss/critic_loss',{str(agent_id):critic_loss},self.global_step+self.n_episodes)
                critic_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                # tensorborad logger: critic parameters
                '''
                self.logger.add_histogram('critic/fc1/weight',self.critic.fc1.weight,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc1/bias',self.critic.fc1.bias,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc2/weight',self.critic.fc2.weight,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc2/bias',self.critic.fc2.bias,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc3/weight',self.critic.fc3.weight,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc3/bias',self.critic.fc3.bias,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc1/weight_grad',self.critic.fc1.weight.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc1/bias_grad',self.critic.fc1.bias.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc2/weight_grad',self.critic.fc2.weight.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc2/bias_grad',self.critic.fc2.bias.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc3/weight_grad',self.critic.fc3.weight.grad,self.global_step+self.n_episodes)
                self.logger.add_histogram('critic/fc3/bias_grad',self.critic.fc3.bias.grad,self.global_step+self.n_episodes)
                '''
                self.critic_optimizer.step()

            
    # predict softmax action based on state
    def _softmax_action(self, state, n_agents):
        state_var = to_tensor_var([state], self.use_cuda)

        softmax_action = []
        for agent_id in range(n_agents):
            softmax_action_var = th.exp(self.actor(state_var[:, agent_id, :]))
            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # choose an action based on state for execution, without softmax
    def action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.argmax(pi))
        return actions

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)

        values = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            value_var = self.critic(state_var[:, agent_id, :], action_var[:, agent_id, :])

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        if is_train:
            env.scene_type = self.env.scene_type
            
        rewards = []  # 各episode的reward
        infos = []  # 各episod各step的信息
        avg_speeds = []  # 各episode的平均速度
        steps = []  # 各episode的step
        flocking_distances = []  # 各episode的平均集群距离
        vehicle_speeds = []  # 各episodes各step各vehicle的速度
        vehicle_positions = []  # 各episode各step各vehicle的位置
        
        self.actor.eval()  # actor设置为eval模式, 固定normalization的更新, 关闭dropout
        video_recorder = None

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            flocking_distance = 0
            vehicle_speed = []
            vehicle_position = []
            reward_i = 0
            info_i = []
            
            done = False
            if is_train:
                state, _ = env.reset(is_training=False, testing_seeds=(self.global_step+self.n_episodes))
            else:
                env.scene_type = i%12
                state, _ = env.reset(is_training=False, testing_seeds=i)
                

            n_agents = len(env.controlled_vehicles)
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                          "testing_episode{}".format(self.n_episodes+self.global_step) + '_{}'.format(i) +
                                          '.mp4')
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                      15))
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=15)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None

            while not done:
                action = self.action(state, n_agents)  # 评价时不用softmax采样
                state, reward, done, info = env.step(action)

                rendered_frame = env.render(mode="rgb_array")
                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)
                # 各step的记录
                avg_speed += info["average_speed"]
                flocking_distance += info['flocking_distance']
                reward_i += reward
                info_i.append(info)
                vehicle_speed.append(info["vehicle_speed"])
                vehicle_position.append(info["vehicle_position"])
                step += 1

            # 各episode的记录
            vehicle_speeds.append(vehicle_speed)
            vehicle_positions.append(vehicle_position)
            rewards.append(reward_i)
            infos.append(info_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)
            flocking_distances.append(flocking_distance / step)

        if video_recorder is not None:
            video_recorder.release()
        env.close()
        
        if is_train:
            self.actor.train()  # 换回train模式
        
        return rewards, steps, avg_speeds, flocking_distances, (vehicle_speeds, vehicle_positions)

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            self.global_step = checkpoint['global_step']
            if train_mode:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                self.actor.train()
                self.critic.train()
            else:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.actor.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        th.save({'global_step': global_step,
                 'actor_state_dict': self.actor.state_dict(),
                 'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                 'critic_state_dict': self.critic.state_dict(),
                 'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                 'actor_target_state_dict': self.actor_target.state_dict(),
                 'critic_target_state_dict':self.critic_target.state_dict()},
                file_path)
