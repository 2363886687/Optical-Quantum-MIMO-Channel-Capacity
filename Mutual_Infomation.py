#功能是调用mutual_info函数 输入光强，返回互信息
#总功率为5 weak下的判决电平为1.00342
#根据输入功率I，得到SISO互信息
#先计算转移概率,p(0)p(1)
#p(0)就是判决电平b以下的pdf累和
import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
from Channel_fading import total_dis_weak
import math

def calculate_entropy(*args):
    # 检查所有概率是否加起来为1
    if not math.isclose(sum(args), 1.0, rel_tol=1e-1):
        raise ValueError("传递的概率之和必须为1")

    # 计算熵
    entropy = -sum(p * math.log2(p) for p in args if p > 0)
    return entropy

def calculate_conditioned_entropy(*args):
    # 计算条件熵H(Y|X)
    entropy = -sum(0.5*p * math.log2(p) for p in args if p > 0)#0.5*p是每个事件发生概率 ，再乘以每个条件概率的信息量
    return entropy
# 计算从负无穷到b的积分
def transfer_pro(total_pdf,pdf0,pdf1,b = 1.00342):
    # p0, error0 = integrate.quad(total_pdf, -np.inf, b, epsabs=1.49e-08, epsrel=1.49e-08)
    # p1, error1 = integrate.quad(total_pdf, b, np.inf, epsabs=1.49e-08, epsrel=1.49e-08)
    # #p1 = 1 - p0
    # print(f"Y接收到0、1的概率为p(0): {p0},p(1) :{p1} ")
    #total_pdf似乎有问题 不计算了
    #只计算 转移概率
    p01,_ = integrate.quad(pdf1,  -np.inf, b)#p(0|1)是对OOK=1条件下接收端从负无穷到b的概率
    p11,_ = integrate.quad(pdf1, b, np.inf)
    p10,_ = integrate.quad(pdf0,  b, np.inf)#p(1|0)是对OOK=0条件下接收端从b到无穷的概率
    p00,_ = integrate.quad(pdf0, -np.inf, b)
    #print(f"p(0|0): {p00},p(1|0) :{p10} ")
    #print(f"p(0|1): {p01},p(1|1) :{p11}")
    p0 = 1/2 *p00 + 1/2 * p01
    p1 = 1/2 *p10 + 1/2 * p11
    #print(f"Y接收到0、1的概率为p(0): {p0},p(1) :{p1} ")
    return p00,p10,p01,p11,p0,p1
def mutual_info(I):
    pdf0, pdf1, total, I0, I1 = total_dis_weak(I)
    total_pdf_fun = interpolate.interp1d(I0, total, bounds_error=False, fill_value=0)
    pdf0_fun = interpolate.interp1d(I0, pdf0, bounds_error=False, fill_value=0)
    pdf1_fun = interpolate.interp1d(I1, pdf1, bounds_error=False, fill_value=0)
    p00, p10, p01, p11, p0, p1 = transfer_pro(total_pdf_fun, pdf0_fun, pdf1_fun)
    Y_entropy = calculate_entropy(p0, p1)
    YX_entropy = calculate_conditioned_entropy(p00, p10, p01, p11)  # 表示P(Y|X)
    #print("Y_entropy:", Y_entropy)
    #print("Y|X_entropy:", YX_entropy)
    mutual_info = Y_entropy - YX_entropy
    #print("mutual_info:",mutual_info)
    return mutual_info

# for I in np.arange(0.1, 5, 0.1):
#     pdf0 ,pdf1, total,I0,I1 = total_dis_weak(I)
#     total_pdf_fun = interpolate.interp1d(I0, total, bounds_error=False, fill_value=0)
#     pdf0_fun = interpolate.interp1d( I0, pdf0, bounds_error=False, fill_value=0)
#     pdf1_fun = interpolate.interp1d( I1, pdf1, bounds_error=False, fill_value=0)
#     p00,p10,p01,p11,p0,p1 = transfer_pro(total_pdf_fun,pdf0_fun,pdf1_fun)
#     Y_entropy = calculate_entropy(p0, p1)
#     YX_entropy = calculate_conditioned_entropy(p00, p10, p01, p11)  # 表示P(Y|X)
#     #print("Y_entropy:", Y_entropy)
#     #print("Y|X_entropy:", YX_entropy)
#     mutual_info = Y_entropy - YX_entropy
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(I1, total, label='total_pdf', color='blue')
#     plt.plot(I0, pdf0, label='pdf0', color='green')
#     plt.plot(I1, pdf1, label='pdf1', color='red')
#     plt.axvline(x=1.00342, color='gray', linestyle='--', label='b=1.00342')
#     plt.title(f'I={I:.2f}_received_distribution_with_poisson_noise_weak_turbulence,b = 1.00342')
#     plt.xlabel('x')
#     plt.ylabel('Probability Density')
#     plt.legend()
#     plt.grid(True)
#     plt.text(0.95, 0.5, f'Mutual Information: {mutual_info:.4f}', transform=plt.gca().transAxes,
#              fontsize=12, verticalalignment='center', horizontalalignment='right')
#     plt.savefig(f'./weak_turbulence/I={I}_received_distribution_with_poisson_noise_,weak_turbulence,b = 1.00342.png')
#     plt.show()
