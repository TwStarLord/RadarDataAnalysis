import numpy as np
from scipy.stats import entropy

echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据

from scipy.fft import fft

# 相对多普勒峰高
doppler_peak = []
for i in range(echo.shape[1]):
    spec = np.abs(fft(echo[:, i]))
    peak = np.max(spec)
    avg = np.mean(spec)
    doppler_peak.append(peak / avg)

print('doppler_peak Len:',len(doppler_peak))
print(doppler_peak)

# 多普勒谱熵
doppler_entropy = []
for i in range(echo.shape[1]):
    spec = np.abs(fft(echo[:, i]))[:echo.shape[0]//2]
    spec /= np.sum(spec)
    doppler_entropy.append(entropy(spec + 1e-12))

print('doppler_entropy Len:',len(doppler_entropy))
print(doppler_entropy)