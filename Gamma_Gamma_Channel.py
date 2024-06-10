import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, kv
import matplotlib
from scipy.integrate import quad

# 设置后端为 'Agg'
matplotlib.use('Agg')

# 定义光强 I 的范围
I = np.linspace(0.01, 5, 1000)

# 计算 Gamma-Gamma 分布的 PDF
def gamma_gamma_pdf(I, alpha, beta):
    nu = alpha - beta
    x = 2 * np.sqrt(alpha * beta * I)
    bessel_k = kv(nu, x)
    pdf = (2 * (alpha * beta)**((alpha + beta) / 2) /
           (gamma(alpha) * gamma(beta)) *
           I**((alpha + beta) / 2 - 1) *
           bessel_k)
    return pdf

# 弱湍流参数
alpha_weak = 11.65
beta_weak = 10.12

# 中湍流参数
alpha_medium = 4.03
beta_medium = 1.91

# 强湍流参数
alpha_strong = 4.23
beta_strong = 1.36

# 计算各自的 PDF 值
pdf_values_weak = gamma_gamma_pdf(I, alpha_weak, beta_weak)
pdf_values_medium = gamma_gamma_pdf(I, alpha_medium, beta_medium)
pdf_values_strong = gamma_gamma_pdf(I, alpha_strong, beta_strong)

# pdf_integral_weak = np.trapz(pdf_values_weak, I)
# # print(f"weak_PDF的积分: {pdf_integral_weak}")
# # pdf_integral_medium = np.trapz(pdf_values_medium, I)
# # print(f"medium_PDF的积分: {pdf_integral_medium}")
# # pdf_integral_strong = np.trapz(pdf_values_strong, I)
# # print(f"strong_PDF的积分: {pdf_integral_strong}")

mean_manual = np.trapz(I * pdf_values_weak, I)
print(f"weak_mean: {mean_manual}")
second_moment = np.trapz((I - mean_manual)**2 * pdf_values_weak, I)
print(f"weak_var: {second_moment}")

# 绘制 PDF 曲线
plt.figure(figsize=(10, 6))
plt.plot(I, pdf_values_weak, label='Weak Turbulence')
plt.plot(I, pdf_values_medium, label='Medium Turbulence')
plt.plot(I, pdf_values_strong, label='Strong Turbulence')
plt.xlabel('Light Intensity I Received')
plt.ylabel('Probability Density Function f(I)')
plt.title('PDF of the Gamma-Gamma Distribution for Different Turbulence Conditions')
plt.legend()
plt.grid(True)

# 保存图像而不是显示
plt.savefig('gamma_gamma_distribution_turbulence.png')
print("The image has been saved as gamma_gamma_distribution_turbulence.png")
