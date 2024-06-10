import torch.cuda

from Env import Env
from MADDPG import MADDPG
import numpy as np
import time
import torch as th
from params import scale_reward
import matplotlib.pyplot as plt
import torch.functional as F
from matplotlib.pyplot import MultipleLocator
import csv
import os
# th.autograd.set_detect_anomaly(True)
# do not render the scene


def reward_plot(reward_list, episode_befor_train,batch_size,action):
    n = len(reward_list)
    x = list(range(len(reward_list)))
    x_axis = x[episode_befor_train:]
    y_axis = []
    for reward_data in reward_list[episode_befor_train:]:
        y_axis.append(reward_data/batch_size)
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.title('reward curve')
    plt.xlim(episode_befor_train, n-1)
    #plt.ylim(4000,4e4)
    plt.plot(x_axis, y_axis)
    plt.text(0.95, 0.5, f'Action:{action}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='right')
    plt.savefig(f"reward_curve.png")
    plt.show()




def save_reward_csv(reward_list, name):
    reward = reward_list
    episode = list(range(len(reward)))
    data_list = []
    for a, b in zip(reward, episode):
        x = {'episode': b, 'reward': a}
        data_list.append(x)

    csv_dir = './reward_csv'
    csv_file_name = name + '.csv'
    csv_path = os.path.join(csv_dir, csv_file_name)
    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['episode', 'reward'])
        for nl in data_list:
            csv_writer.writerow(nl.values())
    f.close()


# G = np.array([[0.207291854034190, 0.692750751349093],
#               [0.983987006627579, 0.832669312809313],
#               [0.978279145873543, 0.721177090945953],
#               [0.178240205307605, 0.553770544093550]])

# G = np.array([[0.992702277, 0.007367785],
#               [0.993524637, 0.983791981],
#               [0.999145893, 0.434490763],
#               [0.987908836, 0.200623389],
#               [0.120590998, 0.999972858],
#               [0.113616881, 0.17931352],
#               [0.041321715, 0.900676289],
#               [0.155035905, 0.979668442]])

np.random.seed(3407)
th.manual_seed(3407)
ns = np.random.randn(4,1) * 0.01

n_agents = 4

M=4
N=4

env = Env( ns=ns,agents=n_agents,seed=3407)
reward_record = []

n_states = np.array([1,1,1,1])
# 前user_num个action为构成W矩阵的智能体，维度对应num_transmitter
# 后面user_num个action为构成F矩阵的智能体, 维度对应num_receiver
#n_actions = np.array([1,1,1,1])
n_actions = np.array([1,1,1,1])
#n_actions = np.array([])

capacity = 50000
batch_size = 256

n_episode = 1000
max_steps = 100
episodes_before_train = 800

param = None
checkpoint = './checkpoint'
name = r"D:\My_research\Channel Capacity Based on RL"

maddpg = MADDPG(n_agents, n_states, n_actions,
                batch_size, capacity,
                episodes_before_train,
                checkpoint_dir=checkpoint,
                name=name)
FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = env.reset()

    obs = np.stack(obs)

    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    total_mse = 0.0
    total_cons = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        # if i_episode % 100 == 0 and e_render:
        #     world.render()
        obs = obs.type(FloatTensor) # 有问题
        # print(obs)
        action = maddpg.select_action(obs, decay=True)  # 有问题
        #action返回归一后的agents * action维度矩阵 返回action维度种策略
        #action = F.softmax(action.contiguous().view(4,16),dim=0)
        #print(action.shape)
        obs_, reward, done, _ = env.step(action)
        reward = th.FloatTensor(reward).type(FloatTensor)  # 有问题
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None
        total_reward += sum(reward.detach().cpu().numpy())/n_agents
        total_mse += env.mse
        total_cons += env.cons1
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs
        c_loss, a_loss = maddpg.update_policy()

    maddpg.episode_done += 1
    print('Episode: %d, reward = %f MSE = %f cons = %f' % (i_episode, total_reward/batch_size, total_mse, total_cons))
    print('action')
    print(env.action)
    reward_record.append(total_reward)


    if i_episode+1 == n_episode:
        maddpg.save_checkpoint()
        reward_plot(reward_record, episodes_before_train,batch_size, env.action)
        save_reward_csv(reward_record, name)

print(env.action)

