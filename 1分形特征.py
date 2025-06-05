import numpy as np

echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据

# Hurst 指数
def hurst_exponent(ts):
    N = len(ts)
    T = np.arange(1, N + 1)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.max(Y) - np.min(Y)
    S = np.std(ts)
    return np.log(R / S) / np.log(N) if S != 0 else 0

hurst_map = np.zeros(echo.shape[1])
for i in range(echo.shape[1]):
    amp_series = np.abs(echo[:, i])
    hurst_map[i] = hurst_exponent(amp_series)

# 这里的Hurst的长度与数据的第二个维度长度保持一致，即与距离单元的个数是保持一致的，说明每个距离单元的数据都会给出一个Hurst指数，
# 该指数与阈值之间将会做一次对比，超过阈值则认为该距离单元 处存在目标
# 现在的问题是，阈值该如何选择
print('Hurst Len:',len(hurst_map))
print(hurst_map)