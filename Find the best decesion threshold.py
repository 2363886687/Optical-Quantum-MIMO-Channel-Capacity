import numpy as np
from scipy.special import gamma, kv
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.integrate import cumtrapz


def gamma_gamma_pdf(I, alpha, beta):
    nu = alpha - beta
    x = 2 * np.sqrt(alpha * beta * I)
    bessel_k = kv(nu, x)
    pdf = (2 * (alpha * beta) ** ((alpha + beta) / 2) /
           (gamma(alpha) * gamma(beta)) *
           I ** ((alpha + beta) / 2 - 1) *
           bessel_k)
    return pdf

def find_intersections(I, pdf1, pdf2):
    diff = pdf1 - pdf2
    sign_change_indices = np.where(np.diff(np.sign(diff)))[0]
    intersections = []
    for idx in sign_change_indices:
        x0, x1 = I[idx], I[idx + 1]
        y0, y1 = diff[idx], diff[idx + 1]
        intersection = x0 - y0 * (x1 - x0) / (y1 - y0)
        intersections.append(intersection)
    return intersections

def find_median(I, combined_pdf):
    cdf = cumtrapz(combined_pdf, I, initial=0)
    median_index = np.where(cdf >= 0.5)[0][0]
    median_value = I[median_index]
    return median_value


def make_received_dis():

    stop=5#表示光强的初始上限，该上限会被I_mean列表转换
    I_range = np.linspace(0.01, stop , 1000)
    b = [[],[],[]]
    for I_high in I_range: # 发射0,1时的发射信号强度数学期望,可调
        I_mean =[0.5,I_high]
        I = I_range  * I_mean[0]

        alpha = [11.65, 4.03, 4.23]
        beta = [10.12, 1.91, 1.36]
        label = ["Weak Turbulence", "Medium Turbulence", "Strong Turbulence"]

        for i in range(3):
            fig, ax1 = plt.subplots(figsize=(10, 6))
            total_combined_pdf = np.zeros_like(I)
            pdf_ook0 = None
            pdf_ook1 = None

            for OOK in range(2):
                I_received_1 = I
                #修改I——received_1的上限，为OOK = 0的时候的光强上限


                pdf = gamma_gamma_pdf(I, alpha[i], beta[i])

                # 计算泊松分布的PMF
                poisson_pmf = poisson.pmf(I_received_1, np.sqrt(I_mean[OOK]))  # 假设泊松分布参数为根号下发出信号的强度

                # 将Gamma-Gamma PDF和泊松PMF相加
                combined_turbulence_pdf = pdf + poisson_pmf

                # 对组合的PDF进行归一化
                combined_turbulence_pdf /= np.trapz(combined_turbulence_pdf, I_received_1)

                if OOK == 0:
                    pdf_ook0 = combined_turbulence_pdf
                else:
                    combined_turbulence_pdf = np.where(I_received_1 > stop *I_mean[0], 0, combined_turbulence_pdf)
                    combined_turbulence_pdf = np.where(I_received_1 == 1, combined_turbulence_pdf / 2,
                                                       combined_turbulence_pdf)

                    pdf_ook1 = combined_turbulence_pdf

                total_combined_pdf += combined_turbulence_pdf / 2

                #ax1.plot(I_received_1, combined_turbulence_pdf, label=f'OOK={OOK} with Poisson Noise')
                #ook=0和ook=1绘制出来的曲线横坐标范围是不一样的 00k=0的横坐标要比=1的短 因为=1的横轴被等比例放大了I_mean倍
                #ook=0的图点更紧密
            intersections = find_intersections(I, pdf_ook0, pdf_ook1)
            b[i].append(intersections)
    for i in range(3):
        average = np.mean(b[i])#假设输入光强均匀分布 输出的判决电平也一定是均匀分布的，求平均
        print(average)


make_received_dis()
