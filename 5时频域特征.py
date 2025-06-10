# import numpy as np
#
# echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据
#
# # 微多普勒特征（均值 + 峰值频率变化）
# from scipy.signal import spectrogram
#
# micro_doppler = []
# for i in range(echo.shape[1]):
#     f, t, Sxx = spectrogram(np.abs(echo[:, i]), fs=1000, nperseg=64)
#     micro_doppler.append(np.var(np.sum(Sxx, axis=1)))  # 多普勒谱形状变化
#
# print('micro_doppler Len:',len(micro_doppler))
# print(micro_doppler)
#
# from skimage.measure import label, regionprops
#
# # 归一化时频图的亮点数和最大连通域
# time_freq_features = []
# for i in range(echo.shape[1]):
#     f, t, Sxx = spectrogram(np.abs(echo[:, i]), fs=1000, nperseg=64)
#     norm_S = Sxx / np.max(Sxx)
#     binary_S = (norm_S > 0.5).astype(int)
#     label_img = label(binary_S)
#     regions = regionprops(label_img)
#     max_area = max([r.area for r in regions]) if regions else 0
#     time_freq_features.append((np.sum(binary_S), max_area))
#
# print('time_freq_features Len:',len(time_freq_features))
# print(time_freq_features)


import numpy as np
from scipy import signal, ndimage


def extract_features(x):
    N = len(x)
    # 1. 全局多普勒功率谱
    F = np.fft.fftshift(np.fft.fft(x, n=N))
    P = np.abs(F) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
    total_power = P.sum()
    # 多普勒质心绝对值
    centroid = (freqs * P).sum() / total_power if total_power > 0 else 0
    doppler_centroid = abs(centroid)
    # 2. 频谱熵
    if total_power > 0:
        P_norm = P / total_power
        spectral_entropy = -np.sum(P_norm[P_norm > 0] * np.log(P_norm[P_norm > 0]))
    else:
        spectral_entropy = 0.0
    # 3. 峰值比（功率峰值除以平均功率）
    P_max = P.max() if len(P) > 0 else 0
    P_mean = total_power / len(P) if total_power > 0 else 0
    peak_to_avg = (P_max / P_mean) if P_mean > 0 else 0
    # 4. STFT时频谱
    f, t, Zxx = signal.stft(x, nperseg=64, noverlap=48, boundary=None, padded=False)
    S = np.abs(Zxx) ** 2
    # 脊线能量累积：各时间帧最大值之和
    ridge_energy = np.sum(S.max(axis=0)) if S.size > 0 else 0
    # 5&6. 连通区域统计
    if S.size > 0:
        thr = 0.25 * S.max()  # 阈值=最大值25%
        mask = S >= thr
        structure = np.ones((3, 3), dtype=int)  # 8邻域结构
        labeled, n_regions = ndimage.label(mask, structure=structure)
        # 最大连通面积
        if n_regions > 0:
            counts = np.bincount(labeled.ravel())[1:]  # 去除背景计数
            max_region_size = counts.max()
        else:
            max_region_size = 0
    else:
        n_regions = 0
        max_region_size = 0
    #  返回：多普勒质心绝对值，频谱熵，峰值比，脊线能量累积，连通区域统计，最大连通面积
    return np.array([doppler_centroid, spectral_entropy, peak_to_avg,
                     ridge_energy, n_regions, max_region_size], dtype=float)


def simulate_clutter(N, rho=0.0, k_shape=None):
    # 生成AR(1)模型的复高斯序列，相关系数rho
    x = np.zeros(N, dtype=np.complex128)
    x[0] = np.random.randn() + 1j * np.random.randn()
    for n in range(1, N):
        e = np.random.randn() + 1j * np.random.randn()
        x[n] = rho * x[n - 1] + np.sqrt(1 - rho ** 2) * e
    # 若指定k_shape，则按K分布调整幅度
    if k_shape is not None:
        u = np.random.gamma(k_shape, 1.0 / k_shape)  # Gamma(shape, scale)取均值1
        x = np.sqrt(u) * x
    return x


# 示例：生成128个脉冲的纯海杂波序列
N = 128
clutter_series = simulate_clutter(N, rho=0.0, k_shape=0.5)


def simulate_target(N, f0=None, f_mod=0.01, f1=None, amp=1.0):
    # 若未指定f0，则随机选取目标基频
    if f0 is None:
        f0_val = np.random.uniform(0.05, 0.2)
        if np.random.rand() < 0.5:
            f0_val = -f0_val  # 随机选择正负方向
        f0 = f0_val
    if f1 is None:
        f1 = 0.02  # 微多普勒调制频率偏移幅度
    phi = 0.0
    y = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        # 每个脉冲更新相位，包含基频f0和微多普勒调制f1*sin(2π f_mod n)
        phi += 2 * np.pi * (f0 + f1 * np.sin(2 * np.pi * f_mod * n))
        y[n] = amp * np.exp(1j * phi)
    return y


# 生成含目标的回波：杂波+目标
clutter_bg = simulate_clutter(N, rho=0.0, k_shape=0.5)
target_echo = simulate_target(N, amp=1.0)
mixed_echo = clutter_bg + target_echo

# 对前述示例计算特征
feat_clutter = extract_features(clutter_series)
feat_target = extract_features(mixed_echo)
# 多普勒质心绝对值，频谱熵，峰值比，脊线能量累积，连通区域统计，最大连通面积
print('多普勒质心绝对值\t频谱熵\t峰值比\t脊线能量累积\t连通区域统计\t最大连通面积')
print("杂波特征:", feat_clutter)
print("含目标特征:", feat_target)
