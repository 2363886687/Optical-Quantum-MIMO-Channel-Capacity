from model import Critic, Actor
import torch
import os
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from torch.optim import SGD
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward
import torch.nn.functional as F
import time


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    """
    不同智能体的动作、状态维度用列表存放
    int: n_agents
    list: dim_obs
    list: dim_act
    """
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, capacity, episodes_before_train, checkpoint_dir, name):
        self.actors = [Actor(dim_obs[i], dim_act[i]) for i in range(n_agents)]

        self.critics = [Critic(n_agents, dim_obs, dim_act, i) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.checkpoint_dir = checkpoint_dir
        self.name = name

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.9
        self.tau = 0.001  # yuanlai 0.01

        #动作噪声是高斯分布,方差
        self.var = [0.2685 for i in range(n_agents)]
        #self.var = [0.4685 for i in range(n_agents)]  # 0.5475      6x2 隐藏层100贼猛 var=0.25  var=0.4685
        # self.var = [0.1 for i in range(n_agents)]
        self.critic_optimizer = [SGD(x.parameters(), lr=0.00001) for x in self.critics]
        self.actor_optimizer = [SGD(x.parameters(), lr=0.00001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        index = 0
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None, batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states_list = [s for s in batch.next_states if s is not None]

            if len(non_final_next_states_list) > 0:
                non_final_next_states = torch.stack(non_final_next_states_list).type(FloatTensor)
            else:
                non_final_next_states = torch.zeros((4, 4, 1), dtype=FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)
            # print(current_Q)

            non_final_next_actions = torch.cat([self.actors_target[i](non_final_next_states[:, i, :]) for i in range(self.n_agents)], dim=1).type(FloatTensor)


            # non_final_next_actions = [
            #     self.actors_target[i](non_final_next_states[:, i, :]) for i in range(self.n_agents)]
            # non_final_next_actions = torch.stack(non_final_next_actions)
            # non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())
            # print(non_final_next_actions.view(-1, sum(self.n_actions)))
            # print(non_final_next_states.view(-1, sum(self.n_states)))
            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask.bool()] = self.critics_target[agent](
                non_final_next_states.view(-1, sum(self.n_states)),
                non_final_next_actions.view(-1, sum(self.n_actions))
            ).squeeze()
            # scale_reward: to scale reward in Q functions
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            loss_Q.backward()
            # print("2:{}".format(torch.cuda.memory_allocated(0)))
            self.critic_optimizer[agent].step()
            # print("3:{}".format(torch.cuda.memory_allocated(0)))
            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i.detach())
            #print(f"action_i shape: {action_i.shape}")
            action_i = self.actors[agent](state_i.detach()).contiguous().view(self.batch_size,1,1)
            ac = action_batch.clone()
            # print(ac)
            # ac[:, agent, :] = action_i
            ac[:, index:index+self.actors[agent].dim_action] = action_i
            index += self.actors[agent].dim_action
            # print(ac)

            whole_action = ac.view(self.batch_size, -1)
            # print(whole_action)
            # print("1:{}".format(torch.cuda.memory_allocated(0) / 1024 ** 2))
            actor_loss = -self.critics[agent](whole_state, whole_action)
            # print("2:{}".format(torch.cuda.memory_allocated(0) / 1024 ** 2))
            actor_loss = actor_loss.mean()
            # print(actor_loss)
            # print("4:{}".format(torch.cuda.memory_allocated(0)))
            actor_loss.backward()
            # print("5:{}".format(torch.cuda.memory_allocated(0)))
            self.actor_optimizer[agent].step()
            # print("6:{}".format(torch.cuda.memory_allocated(0)))
            c_loss.append(loss_Q.item())
            a_loss.append(actor_loss.item())

        if self.steps_done % 300 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch, decay):
        # state_batch: n_agents x state_dim
        """
            list: actions 为所有智能体的动作的叠加，即联合动作

        """
        # actions = torch.zeros(
        #     self.n_agents,
        #     self.n_actions)
        actions = torch.zeros(sum(self.n_actions))

        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        index = 0
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0))  # 有问题

            # 动作加入噪声
            if decay:
                act += torch.from_numpy(
                    np.random.randn(self.n_actions[i]) * self.var[i]).type(FloatTensor)   # 有问题  act = act + ()  改成 act += ()就好了
                # act += torch.from_numpy(
                #     np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)
                if self.episode_done > self.episodes_before_train and\
                   self.var[i] > 0.04:
                    # self.var[i] *= 0.999998
                    self.var[i] *= 0.9999
            act = torch.clamp(act,-1.0, 1.0)  # 有问题

            # actions[i, :] = act
            actions[index:index+self.actors[i].dim_action] = act
            index += self.actors[i].dim_action
        self.steps_done += 1

        actions = F.softmax(actions.contiguous().view(4,1),dim=0)
        actions = actions * 5
        return actions

    def save_checkpoint(self):
        for i in range(self.n_agents):
            actor_name = self.name + '_actor_' + str(i)
            actor_target_name = self.name + '_actor_target_' + str(i)
            critic_name = self.name + '_critic_' + str(i)
            critic_target_name = self.name + '_critic_target_' + str(i)
            torch.save(self.actors[i].state_dict(), os.path.join(self.checkpoint_dir, actor_name))
            torch.save(self.actors_target[i].state_dict(), os.path.join(self.checkpoint_dir, actor_target_name))
            torch.save(self.critics[i].state_dict(), os.path.join(self.checkpoint_dir, critic_name))
            torch.save(self.critics_target[i].state_dict(), os.path.join(self.checkpoint_dir, critic_target_name))

    def load_checkpoint(self):
        for i in range(self.n_agents):
            actor_name = self.name + '_actor_' + str(i)
            actor_target_name = self.name + '_actor_target_' + str(i)
            critic_name = self.name + '_critic_' + str(i)
            critic_target_name = self.name + '_critic_target_' + str(i)
            self.actors[i].load_state_dict(torch.load(os.path.join(self.checkpoint_dir, actor_name)))
            self.actors_target[i].load_state_dict(torch.load(os.path.join(self.checkpoint_dir, actor_target_name)))
            self.critics[i].load_state_dict(torch.load(os.path.join(self.checkpoint_dir, critic_name)))
            self.critics_target[i].load_state_dict(torch.load(os.path.join(self.checkpoint_dir, critic_target_name)))
