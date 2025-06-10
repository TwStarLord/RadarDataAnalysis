# # import numpy as np
# #
# # echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据
# #
# # relative_mean_amp = np.mean(np.abs(echo), axis=0) / np.max(np.mean(np.abs(echo), axis=0))
# #
# # print('entropy_map Len:',relative_mean_amp.shape)
# # print(relative_mean_amp)
# #
# #
# # from scipy.stats import entropy
# #
# # entropy_map = []
# # for i in range(echo.shape[1]):
# #     amp = np.abs(echo[:, i])
# #     hist, _ = np.histogram(amp, bins=100, density=True)
# #     entropy_map.append(entropy(hist + 1e-12))
# #
# # print('entropy_map Len:',len(entropy_map))
# # print(entropy_map)
# #
# #
# # scf_map = []
# # for i in range(echo.shape[1]):
# #     amp = np.abs(echo[:, i])
# #     scf = np.var(amp) / np.mean(amp)**2
# #     scf_map.append(scf)
# #
# #
# # print('scf_map Len:',len(scf_map))
# # print(scf_map)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 参数设置
# N = 1024  # 脉冲数（序列长度）
# r = 4.0  # Logistic映射参量
# x = np.random.rand()  # Logistic初值随机
# amplitude = np.zeros(N)  # 存储幅度序列
#
# # 生成Logistic混沌幅度序列
# for n in range(N):
#     x = r * x * (1 - x)
#     amplitude[n] = x
#
# # 插入目标段
# target_start, target_end = 400, 450  # 目标出现的起止脉冲索引
# boost_factor = 5.0
# amplitude[target_start:target_end] *= boost_factor  # 提升目标段幅度
#
# # 生成对应复数IQ信号（随机相位）
# phase = 2 * np.pi * np.random.rand(N)  # [0,2π)均匀分布相位
# signal = amplitude * np.exp(1j * phase)
#
# # 可视化幅度序列并标注目标段
# plt.figure(figsize=(10, 6))
# plt.plot(np.abs(signal), color='orange', label='Amplitude')
# plt.axvspan(target_start, target_end, color='red', alpha=0.3, label='Target segment')
# plt.xlabel('Pulse index');
# plt.ylabel('Amplitude')
# plt.title('Simulated Sea Clutter with Target segment')
# plt.legend()
# plt.show()
#
# # 相对平均幅度RAA
# # 示例：cut_signal为待测单元复信号，refs为K个参考单元复信号的数组
# cut_signal = signal                      # 这里信号本身作为待测单元示例
# refs = np.vstack([                        # 构造K=5个参考单元（仅杂波，无目标）
#     np.random.permutation(signal)         # 简单将原信号打乱顺序作为模拟参考
#     for _ in range(5)
# ])
#
# # 1. 提取幅度序列
# amp_cut = np.abs(cut_signal)             # 待测单元幅度
# amp_refs = np.abs(refs)                  # 参考单元幅度 (形状K×N)
#
# # 2. 计算平均幅度
# A_cut = amp_cut.mean()
# A_refs = amp_refs.mean(axis=1)          # 每个参考单元的平均幅度 (长度K)
# A_refs_mean = A_refs.mean()             # 参考单元平均幅度的平均
#
# # 3. 计算RAA特征
# RAA_value = A_cut / A_refs_mean
# print(f"RAA = {RAA_value:.3f}")
#
# # 时域信息熵
# # 提取幅度序列（假设 signal 定义同前）
# amp = np.abs(signal)
#
# # 归一化幅度得到概率分布
# p = amp / amp.sum()          # 各幅度点相对总幅度的占比
# # 计算信息熵（过滤掉为0的项以避免 log2(0)）
# p_nonzero = p[p > 1e-12]     # 去除数值上为0的概率项
# entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
# print(f"Information Entropy = {entropy:.3f} bits")
#
#
# # 时域Hurst指数
# # Hurst指数计算函数 (R/S法)
# def hurst_rs(series):
#     series = np.asarray(series, dtype=float)
#     N = series.size
#     if N < 2:
#         return None
#     # 1. 均值中心化
#     Y = series - series.mean()
#     # 2. 累计离差序列
#     Z = np.cumsum(Y)
#     # 3. 计算极差R和标准差S
#     R = Z.max() - Z.min()
#     S = Y.std(ddof=0)
#     # 4. 计算Hurst指数
#     if S < 1e-8 or R < 1e-8:
#         return 0.5  # 序列无波动，H取0.5
#     H = np.log10(R/S) / np.log10(N)
#     return H
#
# # 计算仿真信号和纯杂波参考的Hurst指数
# H_signal = hurst_rs(np.abs(signal))
# H_clutter = hurst_rs(np.abs(refs[0]))   # 取一个纯杂波序列作参考
# print(f"Hurst (with target) = {H_signal:.3f}, Hurst (clutter only) = {H_clutter:.3f}")
#
#
# # 一致性散斑因子
# # 计算相邻脉冲幅度的相关系数作为SCF近似
# amp = np.abs(signal)
# amp_clutter = np.abs(refs[0])  # 取一个无目标杂波序列
#
# # 方法1：直接用numpy的corrcoef计算滞后1相关
# scf = np.corrcoef(amp[:-1], amp[1:])[0,1]
# scf_clutter = np.corrcoef(amp_clutter[:-1], amp_clutter[1:])[0,1]
# print(f"SCF (with target) = {scf:.3f}, SCF (clutter only) = {scf_clutter:.3f}")
# # 方法2：手动计算相关系数
# # 计算幅度序列自相关（归一化）函数
# def autocorr(x, maxlag):
#     x = x - x.mean()
#     result = []
#     var = np.sum(x**2)
#     for lag in range(maxlag+1):
#         if lag == 0:
#             result.append(1.0)
#         else:
#             cov = np.sum(x[:-lag] * x[lag:]) / var
#             result.append(cov)
#     return result
#
# lags = range(0, 11)
# acf = autocorr(amp, maxlag=10)
# acf_clutter = autocorr(amp_clutter, maxlag=10)
# print("Lag 1 autocorr (target) =", acf[1], ", (clutter) =", acf_clutter[1])
#

import numpy as np
# --- 假设 data 为 N_pulses×N_cells 的复数矩阵 ---
# 例：仿真数据生成
N_pulses, N_cells = 1024, 14
r = 4.0
data = np.zeros((N_pulses, N_cells), dtype=complex)
for cell in range(N_cells):
    x = np.random.rand()
    amp = np.zeros(N_pulses)
    for n in range(N_pulses):
        x = r * x * (1 - x)
        amp[n] = x
    if cell == 7:  # 在第8个单元插入目标
        amp[400:450] *= 5.0
    phase = 2 * np.pi * np.random.rand(N_pulses)
    data[:, cell] = amp * np.exp(1j * phase)

# --- 特征函数 ---
def compute_raa(data, j, K=6):
    """计算第 j 列的 RAA"""
    N_cells = data.shape[1]
    left = max(0, j - K//2)
    right = min(N_cells, j + K//2 + 1)
    refs = [i for i in range(left, right) if i != j]
    amp_j = np.abs(data[:, j])
    A_j = amp_j.mean()
    A_refs = np.abs(data[:, refs]).mean(axis=0)
    return A_j / A_refs.mean()

def compute_entropy(amp):
    """香农信息熵"""
    p = amp / amp.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def compute_hurst(amp):
    """R/S 方法估计 Hurst 指数"""
    y = amp - amp.mean(); z = np.cumsum(y)
    R = z.max() - z.min(); S = y.std(ddof=0)
    if S < 1e-8 or R < 1e-8: return 0.5
    return np.log10(R/S) / np.log10(len(amp))

def compute_scf(amp):
    """滞后1自相关系数"""
    return np.corrcoef(amp[:-1], amp[1:])[0,1]

# --- 批量计算并输出 ---
RAA = np.zeros(N_cells); Ent = np.zeros(N_cells)
Hurst = np.zeros(N_cells); SCF = np.zeros(N_cells)
for j in range(N_cells):
    amp_j = np.abs(data[:, j])
    RAA[j]   = compute_raa(data, j)
    Ent[j]   = compute_entropy(amp_j)
    Hurst[j] = compute_hurst(amp_j)
    SCF[j]   = compute_scf(amp_j)

print('RAA:',   np.round(RAA,3))
print('Entropy:', np.round(Ent,3))
print('Hurst:',  np.round(Hurst,3))
print('SCF:',    np.round(SCF,3))