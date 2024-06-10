import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
from Channel_fading import total_dis_weak
from Mutual_Infomation import transfer_pro,calculate_entropy,calculate_conditioned_entropy
for I in np.arange(0.1, 5.1, 0.1):
    pdf0 ,pdf1, total,I0,I1 = total_dis_weak(I)
    total_pdf_fun = interpolate.interp1d(I0, total, bounds_error=False, fill_value=0)
    pdf0_fun = interpolate.interp1d( I0, pdf0, bounds_error=False, fill_value=0)
    pdf1_fun = interpolate.interp1d( I1, pdf1, bounds_error=False, fill_value=0)
    p00,p10,p01,p11,p0,p1 = transfer_pro(total_pdf_fun,pdf0_fun,pdf1_fun)
    Y_entropy = calculate_entropy(p0, p1)
    YX_entropy = calculate_conditioned_entropy(p00, p10, p01, p11)  # 表示P(Y|X)
    #print("Y_entropy:", Y_entropy)
    #print("Y|X_entropy:", YX_entropy)
    mutual_info = Y_entropy - YX_entropy

    plt.figure(figsize=(10, 6))
    plt.plot(I1, total, label='total_pdf', color='blue')
    plt.plot(I0, pdf0, label='pdf0', color='green')
    plt.plot(I1, pdf1, label='pdf1', color='red')
    plt.axvline(x=1.00342, color='gray', linestyle='--', label='b=1.00342')
    plt.title(f'I={I:.2f}_received_distribution_with_poisson_noise_weak_turbulence,b = 1.00342')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.text(0.95, 0.5, f'Mutual Information: {mutual_info:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='right')
    plt.savefig(f'./weak_turbulence/I={I:.2f}_received_distribution_with_poisson_noise_,weak_turbulence,b = 1.00342.png')
    plt.show()