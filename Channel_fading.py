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


def total_dis_weak(I_power):
    stop = 5
    I = np.linspace(0.01,stop, 1000)
    alpha = 11.65
    beta = 10.12
    I_mean = [0.5, I_power]  # 发射0,1时的发射信号强度数学期望,可调
    total_combined_pdf = np.zeros_like(I)
    pdf0 =[]
    pdf1 =[]
    I0 = []
    I1 = []
    for OOK in range(2):

        I_received_1 = I * I_mean[OOK]

        pdf = gamma_gamma_pdf(I, alpha, beta)

        # 计算泊松分布的PMF
        poisson_pmf = poisson.pmf(I_received_1, np.sqrt(I_mean[OOK]))  # 假设泊松分布参数为根号下发出信号的强度

        # 将Gamma-Gamma PDF和泊松PMF相加
        combined_turbulence_pdf = pdf + poisson_pmf

        # 对组合的PDF进行归一化
        combined_turbulence_pdf /= np.trapz(combined_turbulence_pdf, I_received_1)

        if OOK == 0:
            pdf0 = combined_turbulence_pdf
            I0 = I_received_1
        else:
            pdf1 = combined_turbulence_pdf
            I1 = I_received_1

        total_combined_pdf += combined_turbulence_pdf / 2

    return pdf0, pdf1, total_combined_pdf,I0,I1




def make_received_dis():
    stop=5#表示光强的初始上限，该上限会被I_mean列表转换
    I = np.linspace(0.01, stop , 1000)
    I_mean = [0.5, 2.0]  # 发射0,1时的发射信号强度数学期望,可调

    alpha = [11.65, 4.03, 4.23]
    beta = [10.12, 1.91, 1.36]
    label = ["Weak Turbulence", "Medium Turbulence", "Strong Turbulence"]

    for i in range(3):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        total_combined_pdf = np.zeros_like(I)
        pdf_ook0 = None
        pdf_ook1 = None

        for OOK in range(2):
            I_received_1 = I * I_mean[OOK]

            pdf = gamma_gamma_pdf(I, alpha[i], beta[i])

            # 计算泊松分布的PMF
            poisson_pmf = poisson.pmf(I_received_1, np.sqrt(I_mean[OOK]))  # 假设泊松分布参数为根号下发出信号的强度

            # 将Gamma-Gamma PDF和泊松PMF相加
            combined_turbulence_pdf = pdf + poisson_pmf

            # 对组合的PDF进行归一化
            combined_turbulence_pdf /= np.trapz(combined_turbulence_pdf, I_received_1)

            total_combined_pdf += combined_turbulence_pdf / 2

            ax1.plot(I_received_1, combined_turbulence_pdf, label=f'OOK={OOK} with Poisson Noise')
            #ook=0和ook=1绘制出来的曲线横坐标范围是不一样的 00k=0的横坐标要比=1的短 因为=1的横轴被等比例放大了I_mean倍
            #ook=0的图点更紧密
        #intersections = find_intersections(I, pdf_ook0, pdf_ook1)
        #for intersection in intersections:
         #   ax1.axvline(intersection, color='r', linestyle='--', label=f'Intersection at {intersection:.4f}')
        ax1.plot(I, total_combined_pdf, label='Total Combined PDF', color='red', linestyle='-', linewidth=2)
        ax1.set_xlabel('Light Intensity I Received')
        ax1.set_ylabel('Probability Density')
        ax1.grid(True)

        # 创建右侧的坐标轴并绘制泊松分布的 PMF 曲线
        ax2 = ax1.twinx()
        ax2.plot(I_received_1, poisson_pmf, 'g--', label='Poisson Noise')
        ax2.set_ylabel('Poisson PMF')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title(f'PDF of I received with poisson noise under {label[i]}')
        plt.savefig(f'I_received_distribution_with_poisson_noise_{label[i]}.png')
        plt.show()

        # 计算总分布的中位数
        #median_value = find_median(I, total_combined_pdf)
        #print(f"{label[i]} 状态下总分布的中位数为: {median_value:.4f}")

        # 绘制 total_combined_pdf 曲线
        # plt.figure(figsize=(10, 6))
        # plt.plot(I, total_combined_pdf, label=f'Total Combined PDF under {label[i]}')
        # #plt.axvline(median_value, color='r', linestyle='--', label=f'Median = {median_value:.4f}')
        # plt.xlabel('Light Intensity I Received')
        # plt.ylabel('Probability Density')
        # plt.title(f'Total Combined PDF under {label[i]}')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f'Total_combined_PDF_{label[i]}.png')
        # plt.show()


#make_received_dis()
