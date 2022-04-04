import random
from collections import namedtuple


Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "masks"))


class OnPolicyReplayMemory(object):
    """
    Replay memory buffer
    每次取出batch后自动reset
    self.memory为具名数组列表
    """
    def __init__(self, capacity,reward_gamma):
        self.capacity = capacity
        self.gamma = reward_gamma
        self.memory = []  # 具名元组Experience的列表,改成放在连续的内存里的数据结构会更快:https://zhuanlan.zhihu.com/p/103605702
        self.position = 0

    def _push_one(self,state,action,reward,next_state,done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 在列表中插入一个None，表示增加长度但是暂时未赋值
        mask = 0.0 if done else self.gamma
        self.memory[self.position] = Experience(state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            # for state,action,reward,next_state,done in zip(states, actions, rewards, next_states, dones):
            #     self._push_one(state,action,reward,next_state,done)
            for transition in zip(states, actions, rewards, next_states, dones):
                self._push_one(*transition)
            '''
            if next_states is not None and len(next_states) > 0:
                for s,a,r,n_s,d in zip(states, actions, rewards, next_states, dones):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s,a,r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
            '''
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self,batch_size):
        # if batch_size > len(self.memory):
        #     batch_size = len(self.memory)
        # transitions = random.sample(self.memory, batch_size)   # memory随机采样
        batch = Experience(*zip(*self.memory))  # memory全部取出
        self.reset()  # sample后马上reset
        return batch

    def __len__(self):
        return len(self.memory)
    
    def reset(self):
        # reset the memory
        self.memory = []
        self.position = 0


class ReplayMemory(object):
    """
    Replay memory buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        return batch

    def __len__(self):
        return len(self.memory)