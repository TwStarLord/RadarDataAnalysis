import numpy as np

echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据

# 微多普勒特征（均值 + 峰值频率变化）
from scipy.signal import spectrogram

micro_doppler = []
for i in range(echo.shape[1]):
    f, t, Sxx = spectrogram(np.abs(echo[:, i]), fs=1000, nperseg=64)
    micro_doppler.append(np.var(np.sum(Sxx, axis=1)))  # 多普勒谱形状变化

print('micro_doppler Len:',len(micro_doppler))
print(micro_doppler)

from skimage.measure import label, regionprops

# 归一化时频图的亮点数和最大连通域
time_freq_features = []
for i in range(echo.shape[1]):
    f, t, Sxx = spectrogram(np.abs(echo[:, i]), fs=1000, nperseg=64)
    norm_S = Sxx / np.max(Sxx)
    binary_S = (norm_S > 0.5).astype(int)
    label_img = label(binary_S)
    regions = regionprops(label_img)
    max_area = max([r.area for r in regions]) if regions else 0
    time_freq_features.append((np.sum(binary_S), max_area))

print('time_freq_features Len:',len(time_freq_features))
print(time_freq_features)