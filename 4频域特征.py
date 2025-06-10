# import numpy as np
# from scipy.stats import entropy
#
# echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据
#
# from scipy.fft import fft
#
# # 相对多普勒峰高
# doppler_peak = []
# for i in range(echo.shape[1]):
#     spec = np.abs(fft(echo[:, i]))
#     peak = np.max(spec)
#     avg = np.mean(spec)
#     doppler_peak.append(peak / avg)
#
# print('doppler_peak Len:',len(doppler_peak))
# print(doppler_peak)
#
# # 多普勒谱熵
# doppler_entropy = []
# for i in range(echo.shape[1]):
#     spec = np.abs(fft(echo[:, i]))[:echo.shape[0]//2]
#     spec /= np.sum(spec)
#     doppler_entropy.append(entropy(spec + 1e-12))
#
# print('doppler_entropy Len:',len(doppler_entropy))
# print(doppler_entropy)

# 相对多普勒峰高 (Relative Doppler Peak Height, RDPH)
# 特征原理： 相对多普勒峰高表示目标单元多普勒谱主峰高度相对于周围杂波背景平均峰高的比值
# 在实际IPIX数据实验中，含目标单元的RDPH通常显著高于1。例如，某次实验中目标单元RDPH达到约30以上，
# 而纯杂波单元的RDPH在1左右。这表明目标的频谱峰值远高于背景平均水平，RDPH对目标存在具有很强的指示性。
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载或生成 IPIX 数据（示例）
data = np.random.randn(131012, 14) + 1j * np.random.randn(131012, 14)

# 2. 提取 CUT 和参考门
CUT_idx = 6                        # Python 0-based index 列7
x = data[:, CUT_idx]              # CUT 序列，形状 (131012,)
# 选取邻近参考门 (第5,6,8,9列)
refs = data[:, [CUT_idx-2, CUT_idx-1, CUT_idx+1, CUT_idx+2]]
refs_list = [refs[:, i] for i in range(refs.shape[1])]

# 3. 公共预处理和函数定义
P = len(x)
w = np.hamming(P)

def compute_fft_power(sig):
    X = np.fft.fft(sig * w)
    return np.abs(X)**2

def compute_rdph(x, refs_list):
    P_cut = compute_fft_power(x)
    peak_cut = P_cut.max()
    ref_peaks = [compute_fft_power(r).max() for r in refs_list]
    return peak_cut / np.mean(ref_peaks)

def compute_rve(x, refs_list):
    P_cut = compute_fft_power(x)
    p_cut = P_cut / (P_cut.sum() + 1e-12)
    H_cut = -np.sum(p_cut * np.log(p_cut + 1e-12))
    H_refs = []
    for r in refs_list:
        P_ref = compute_fft_power(r)
        p_ref = P_ref / (P_ref.sum() + 1e-12)
        H_refs.append(-np.sum(p_ref * np.log(p_ref + 1e-12)))
    return H_cut / np.mean(H_refs)

def compute_fpar(x):
    P_cut = compute_fft_power(x)
    return P_cut.max() / P_cut.mean()

def compute_hurst(x):
    P_cut = compute_fft_power(x)
    N = len(P_cut)
    half = N // 2
    f = np.arange(1, half) / N
    PSD = P_cut[1:half]
    slope, _ = np.polyfit(np.log(f), np.log(PSD + 1e-12), 1)
    beta = -slope
    return (beta + 1) / 2

# 4. 计算各特征
RDPH_val = compute_rdph(x, refs_list)
RVE_val = compute_rve(x, refs_list)
FPAR_val = compute_fpar(x)
Hurst_val = compute_hurst(x)
print(f"RDPH = {RDPH_val:.3f}, RVE = {RVE_val:.3f}, FPAR = {FPAR_val:.3f}, Hurst = {Hurst_val:.3f}")

# 5. 绘制耦合波 vs 纯海杂波功率谱对比
# 定义耦合波和纯杂波示例
x_coupled = data[:, CUT_idx]                # 含目标回波
clutter_idx = CUT_idx - 5                   # 选取一个纯杂波门，例如第2列
x_clutter = data[:, clutter_idx]            # 纯海杂波回波

# 计算正频谱功率
P_c = compute_fft_power(x_coupled)[:P//2]
P_l = compute_fft_power(x_clutter)[:P//2]

freq = np.linspace(0, 0.5, P//2)           # 归一化多普勒频率
plt.figure(figsize=(8,4))
plt.plot(freq, 10*np.log10(P_l), label='纯海杂波')
plt.plot(freq, 10*np.log10(P_c), label='耦合波（含目标）')
plt.xlabel('归一化多普勒频率')
plt.ylabel('功率谱 (dB)')
plt.title('耦合波 vs 纯海杂波功率谱')
plt.legend()
plt.grid(True)
plt.show()