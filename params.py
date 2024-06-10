import torch
scale_reward = 0.01
step_max = 1000
planck_constant = 6.62607015*1e-34
righthand = 0.06 * -188.18 / (planck_constant * 680*1e6)
# print(righthand)
# x = 10**(-155/10)
# print(0.06*x/(planck_constant * 680*1e6))
# print(0.06*x/(planck_constant*3*1e8/(266*1e-9)))
# a = torch.zeros((2,16))
# # a = torch.cat([a], dim=1)
# b = torch.tensor([[0,0,1,3],[3,4,5,6]])
# a[:, 0:4] = b
# print(a)
# c = torch.cat([a,b],dim=1)
# print(c)

