"""
分形特征
"""
import numpy as np

echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据

# # Hurst 指数
# def hurst_exponent(ts):
#     N = len(ts)
#     T = np.arange(1, N + 1)
#     Y = np.cumsum(ts - np.mean(ts))
#     R = np.max(Y) - np.min(Y)
#     S = np.std(ts)
#     return np.log(R / S) / np.log(N) if S != 0 else 0
#
# # 论文中的Hurst的x(n)代表什么，是快时间维度还是慢时间维度，此处代表的是慢时间维度的脉冲维时间序列
# def hurst_exponent(ts):
#     F_m = []
#     N = len(ts)
#     T = np.arange(1, N + 1)
#     Y = np.cumsum(ts - np.mean(ts))
#     R = np.max(Y) - np.min(Y)
#     S = np.std(ts)
#     return np.log(R / S) / np.log(N) if S != 0 else 0
#
#
# hurst_map = np.zeros(echo.shape[1])
# for i in range(echo.shape[1]):
#     amp_series = np.abs(echo[:, i])
#     hurst_map[i] = hurst_exponent(amp_series)
#
# # 这里的Hurst的长度与数据的第二个维度长度保持一致，即与距离单元的个数是保持一致的，说明每个距离单元的数据都会给出一个Hurst指数，
# # 该指数与阈值之间将会做一次对比，超过阈值则认为该距离单元 处存在目标
# # 现在的问题是，阈值该如何选择
# print('Hurst Len:',len(hurst_map))
# print(hurst_map)


import numpy as np
import matplotlib.pyplot as plt


def hurst_exponent(x, min_scale=2, max_scale=None, scale_ratio=2):
    """
    计算序列 x 的 Hurst 指数（q=2）。

    参数:
    - x: 一维 NumPy 数组，表示某距离单元上的回波幅度序列 x(n)。后续修改为二维（N,D），N为脉冲数，D为距离单元数
    - min_scale: 最小窗口长度 m，默认为 8。
    - max_scale: 最大窗口长度 m，如果为 None，则自动取 N // 4。
    - scale_ratio: 尺度倍增因子，默认为 2，即 m = min_scale * (scale_ratio ** k)。

    返回:
    - H: Hurst 指数。
    - scales: 实际使用的尺度列表。
    - F2: 对应尺度下的 F^{(2)}(m) 值列表。
    """
    # 1. 去均值，构造随机游走序列 y
    x = np.asarray(x, dtype=np.float64)
    # 这里参考论文，序列长度为917504，对应PRF2000为327.68s
    N, D = x.shape[0], x.shape[1]
    mu = np.mean(x, axis=0)
    y = np.cumsum(x - mu, axis=0)  # 随机游走序列 y(k)

    # 2. 确定尺度列表
    if max_scale is None:
        # max_scale = N // 4
        max_scale = N
    m = min_scale
    scales = []
    while m <= max_scale:
        scales.append(int(m))
        m *= scale_ratio
    scales = np.array(scales, dtype=int)

    # 3. 对每个尺度 m 计算 F^{(2)}(m)
    F2 = []
    for dis_idx in range(D):
        F2_dis = []
        y = x[:, dis_idx]
        for m in scales:
            # 计算所有 n 范围内的增量
            # y[n+m] - y[n], 有效 n 范围为 [0, N - m - 1]
            diffs = y[m:] - y[:-m]
            # 计算均方根
            F2_m = np.sqrt(np.mean(diffs ** 2))
            F2_dis.append(F2_m)
        F2_dis = np.array(F2_dis)
        F2.append(F2_dis)
    F2 = np.array(F2)
    # 4. 双对数线性拟合
    log_scales = np.log2(scales)
    log_F2 = np.log2(F2)
    # 仅选取靠近线性区间的数据，一般可以选取除第一个和最后一个之外的
    # 这里为了简化，选取所有点进行拟合
    Hursts = []
    for dis_idx in range(D):
        slope, intercept = np.polyfit(log_scales, log_F2[dis_idx, :], 1)
        H = slope
        Hursts.append(H)

    return Hursts, scales, F2


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    save_path = 'F:/毕业相关/ZXY/毕业论文/第二章/多维特征/Hurst指数/'
    # 假设我们已经从雷达数据中提取了某个距离单元的回波幅度序列 x(n)
    # 这里以一个模拟随机分形序列为例
    np.random.seed(42)
    N = 2 ** 16
    D = 14
    # 生成一个伪随机分形信号示例（并不是严格真实的海杂波，仅作演示）
    x = np.cumsum(np.random.randn(N, D), axis=0)

    # 计算 Hurst 指数
    H, scales, F2 = hurst_exponent(x, min_scale=2)
    # print(f"Estimated Hurst exponent: {H:.4f}")

    # 3. 绘制多组 F2 曲线，并标注拟合直线
    plt.figure(figsize=(8, 6))
    log_scales = np.log2(scales)
    for idx in range(D):
        log_F2 = np.log2(F2[idx])
        # 绘制每个距离单元的点线
        plt.plot(log_scales, log_F2, '-o', label=f'距离单元 {idx + 1}：H={H[idx]:.2f}')
        # 绘制该单元的拟合直线
        coef = np.polyfit(log_scales, log_F2, 1)
        fit_line = coef[0] * log_scales + coef[1]
        plt.plot(log_scales, fit_line, '--')

    # 4. 添加坐标标签、标题和图例（中文）
    plt.xlabel('log₂ m')
    plt.ylabel('log₂ F^(2)(m)')
    plt.title('14 个距离单元双对数曲线与 Hurst 指数对比', )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_path}hurst_exponent.png',dpi=300)

    H = np.random.randn(D)
    # 可视化Hurst指数
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(1,D+1)), H)
    plt.xlabel('距离单元')
    plt.ylabel('Hurst指数')
    plt.title('Hurst指数随距离单元变化')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{save_path}hurst_exponent_change.png',dpi=300)
