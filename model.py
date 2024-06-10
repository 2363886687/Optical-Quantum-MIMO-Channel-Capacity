import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Critic(nn.Module):
    """
    各个智能体的动作维度、状态维度不一样,
    因此传入的数据结构是列表，
    里面存放着不同智能体的动作、状态维度
    int: n_agent
    list: dim_observation
    list: dim_action
    int: id
    """
    def __init__(self, n_agent, dim_observation, dim_action, id):
        super().__init__()
        self.n_agent = n_agent
        # self.dim_observation = dim_observation
        # self.dim_action = dim_action
        # obs_dim = dim_observation * n_agent
        # act_dim = self.dim_action * n_agent
        self.first_layer_dim = sum(dim_observation) + sum(dim_action)
        self.dim_observation = dim_observation[id]
        self.dim_action = dim_action[id]
        # self.fc1 = nn.Linear(self.first_layer_dim, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 1)

        self.fc1 = nn.Linear(self.first_layer_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        # self.fc4 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, obs, acts):
        state = torch.cat((obs,), dim=1)           # 所有智能体联合观测
        # action = torch.cat(acts, dim=1)         # 所有智能体联合行为
        x = torch.cat([state, acts], dim=1)  # 联合观测 + 联合行为
        x = F.relu(self.fc1(x).detach())
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        q_value = self.fc4(x)
        return q_value


class Actor(nn.Module):#每个actor负责自己的功率分配
    """
    各个智能体的动作维度、状态维度不一样,
    因此传入的数据结构是列表，
    里面存放着不同智能体的动作、状态维度
    int: n_agent        4个
    list: dim_observation       16个
    list: dim_action

    """
    def __init__(self, dim_observation, dim_action):
        super().__init__()
        self.dim_action = dim_action
        self.dim_observation = dim_observation
        # self.fc1 = nn.Linear(dim_observation, 128)  # 定义输入观测维度
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, dim_action)      # 定义输出动作维度
        self.fc1 = nn.Linear(dim_observation, 200)  # 定义输入观测维度
        self.fc2 = nn.Linear(200, 200)
        # self.fc3 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, dim_action)      # 定义输出动作维度

    def forward(self, obs):
        # 动作输出范围在0到1
        # 但是公式推导中的预编码矩阵中的元素是实数域
        result = F.relu(self.fc1(obs))
        result = F.relu(self.fc2(result))
        # result = F.relu(self.fc3(result))
        result = F.relu(self.fc3(result))
        result = self.fc4(result)
        #result = F.relu(result)
        result = torch.sigmoid(result)  # 使用 sigmoid 将输出压缩到 0 到 1 之间

        return result
