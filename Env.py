import time
import torch
import numpy as np
import copy
from params import step_max
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

import torch.nn.functional as F
import numpy as np
from Mutual_Infomation import mutual_info

class Env:
    def __init__(self,ns,agents, seed,M = 4, N = 4, t = 25):
        super().__init__()
        np.random.seed(seed)
        self.M = M
        self.N = N
        self.t = t
        self.ns = ns
        # self.init_random_action = np.random.randn(n_w_agents + n_f_agents, num_transmitter, num_receiver)
        self.agents = agents
        self.step_max = step_max

        # 环境的状态设置为[W,F]两个矩阵
        self.state = np.zeros((agents, 1))
        # 计算得到背景光子数以及最大光子数
        self.mse = 0
        self.cons1 = 0
        #random_action = np.random.randn(self.agents*2**M)*0.0001

        self.n_actions = 1
        mean = (0.5 + 5.0) / 2  # 正态分布的均值，取区间的中间值
        std = (5.0 - 0.5) / 6  # 正态分布的标准差，通常设置为区间长度的六分之一
        lower_bound = 0.5
        upper_bound = 5.0

        # 生成随机动作
        random_actions = np.random.normal(loc=mean, scale=std, size=(agents, self.n_actions))

        # 将动作限制在 0.5 到 5.0 之间
        random_action = np.clip(random_actions, lower_bound, upper_bound)
        random_action_tensor = torch.tensor(random_action)
        actions = F.softmax(random_action_tensor.contiguous().view(4,1),dim=0)
        actions = actions * 5#归一化并放到5 总功率为5
        print("初始随机动作：",actions)

        self.init_random_action = actions
        self.action = copy.deepcopy(self.init_random_action)


    def reset(self):#初始设定 返回空状态列表4*1
        self.action = copy.deepcopy(self.init_random_action)
        self.step_max = step_max
        self.reward()
        return self.state

    def step(self, action):
        s = self.state
        self.action = action.detach().cpu().numpy()
        r = self.reward()
        s_ = self.state

        self.step_max = self.step_max - 1
        done = False
        # 默认为False
        if self.step_max == 0:
            done = True
        return s_, r, done, s

    def reward(self):
        def I(agents):
            MIMO_INFO = 0
            for p in agents:#agents是传入的列表 1*4 p表示每个agent的功率
                MIMO_INFO = MIMO_INFO + mutual_info(p)
            #print("MIMO_INFO:",MIMO_INFO)

            return MIMO_INFO

        flattened_list = self.action.flatten().tolist()

        I_val = I(flattened_list)

        return np.array([I_val, I_val, I_val, I_val])