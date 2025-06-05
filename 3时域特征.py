import numpy as np

echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据

relative_mean_amp = np.mean(np.abs(echo), axis=0) / np.max(np.mean(np.abs(echo), axis=0))

print('entropy_map Len:',relative_mean_amp.shape)
print(relative_mean_amp)


from scipy.stats import entropy

entropy_map = []
for i in range(echo.shape[1]):
    amp = np.abs(echo[:, i])
    hist, _ = np.histogram(amp, bins=100, density=True)
    entropy_map.append(entropy(hist + 1e-12))

print('entropy_map Len:',len(entropy_map))
print(entropy_map)


scf_map = []
for i in range(echo.shape[1]):
    amp = np.abs(echo[:, i])
    scf = np.var(amp) / np.mean(amp)**2
    scf_map.append(scf)


print('scf_map Len:',len(scf_map))
print(scf_map)