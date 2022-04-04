import torch as th
from torch import nn

class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, hidden_size, output_size,
                 config=None):
        super(ActorNetwork, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        if self.config['activation_type']=='ReLU':
            self.activator = nn.functional.relu
        elif self.config['activation_type']=='tanh':
            self.activator = th.tanh
        else:
            raise ValueError('Invalid activation type')

        # Layer normalization
        if self.config['use_layer_norm']:
            # LayerNorm: element-wise affine
            # BatchNorm: channel-wise affine
            self.ln1 = nn.LayerNorm(hidden_size,elementwise_affine=True)
            self.ln2 = nn.LayerNorm(hidden_size,elementwise_affine=True)
            
        # weight正交初始化和bias常数初始化
        if self.config['use_orthogonal_init']:
            for module in self.modules():
                if isinstance(module,nn.Linear): 
                    nn.init.orthogonal_(module.weight,1)
                    nn.init.constant_(module.bias,0)


    def __call__(self, state):
        out = self.activator(self.fc1(state))
        if self.config['use_layer_norm']:
            out = self.ln1(out) 
        out = self.activator(self.fc2(out))
        if self.config['use_layer_norm']:
            out = self.ln2(out)
        out = nn.functional.log_softmax(self.fc3(out),dim=-1)
        return out

class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, hidden_size, output_size=1,
                config=None):
        super(CriticNetwork, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        if self.config['activation_type']=='ReLU':
            self.activator = nn.functional.relu
        elif self.config['activation_type']=='tanh':
            self.activator = th.tanh
        else:
            Exception('Invalid activation type.')
            
        # Layer normalization
        if self.config['use_layer_norm']:
            self.ln1 = nn.LayerNorm(hidden_size,elementwise_affine=True)
            self.ln2 = nn.LayerNorm(hidden_size,elementwise_affine=True)
            
        # weight正交初始化和bias常数初始化
        if self.config['use_orthogonal_init']:
            for module in self.modules():
                if isinstance(module,nn.Linear): 
                    nn.init.orthogonal_(module.weight,1)
                    nn.init.constant_(module.bias,0)

    def __call__(self, state):
        out = self.activator(self.fc1(state))
        if self.config['use_layer_norm']:
            out = self.ln1(out) 
        out = self.activator(self.fc2(out))
        if self.config['use_layer_norm']:
            out = self.ln2(out) 
        out = self.fc3(out)
        return out

class ActorNetworkPlus(nn.Module):
    def __init__(self,obs_dim,shared_size,hidden_size,action_dim,config=None):
        super(ActorNetworkPlus,self).__init__()
        self.obs_dim = obs_dim
        self.config = config
        
        if self.config['activation_type']=='ReLU':
            self.activator = nn.ReLU()
        elif self.config['activation_type']=='tanh':
            self.activator = nn.Tanh()
        else:
            raise ValueError('Invalid activation type')
        
        self.fc_shared_1 = nn.Linear(obs_dim,shared_size)
        self.fc_shared_2 = nn.Linear(shared_size,hidden_size)
        self.shared_block = nn.Sequential(self.fc_shared_1,
                                          self.activator,
                                          self.fc_shared_2,
                                          self.activator)
        self.fc_hidden_1 = nn.Linear(hidden_size,hidden_size)
        self.fc_hidden_2 = nn.Linear(hidden_size,action_dim)
        self.hidden_block = nn.Sequential(self.fc_hidden_1,
                                          self.activator,
                                          self.fc_hidden_2,
                                          nn.LogSoftmax(dim=-1))
    
    def __call__(self,state):
        state_var = state.view(-1,5,self.obs_dim)  # 第二个参数为观测到的总车辆数(包括自身)
        out = th.sum(self.shared_block(state_var),dim=1)
        out = self.hidden_block(out)   
        return  out
    
class CriticNetworkPlus(nn.Module):
    def __init__(self,obs_dim,shared_size,hidden_size,config=None):
        super(CriticNetworkPlus,self).__init__()
        self.obs_dim = obs_dim
        self.config = config

        if self.config['activation_type']=='ReLU':
            self.activator = nn.ReLU()
        elif self.config['activation_type']=='tanh':
            self.activator = nn.Tanh()
        else:
            raise ValueError('Invalid activation type')
        
        self.fc_shared_1 = nn.Linear(obs_dim,shared_size)
        self.fc_shared_2 = nn.Linear(shared_size,hidden_size)
        self.shared_block = nn.Sequential(self.fc_shared_1,
                                          self.activator,
                                          self.fc_shared_2,
                                          self.activator)
        self.fc_hidden_1 = nn.Linear(hidden_size,hidden_size)
        self.fc_hidden_2 = nn.Linear(hidden_size,1)
        self.hidden_block = nn.Sequential(self.fc_hidden_1,
                                          self.activator,
                                          self.fc_hidden_2,
                                          self.activator)
        
    def __call__(self,state):
        state_var = state.view(-1,5,self.obs_dim)  # 第二个参数为观测到的总车辆数(包括自身)
        out = th.sum(self.shared_block(state_var),dim=1)
        return self.hidden_block(out)    
    
# 共用底层的Actor-CriticNetwork，未使用
class ActorCriticNetwork(nn.Module):
    """
    An actor-critic network that shared lower-layer representations but
    have distinct output layers
    """

    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=1):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val
