import numpy as np

echo_HH = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据
echo_HV = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据
echo_VH = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据
echo_VV = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据

# 假设有四个通道数据：HH, VV, HV, VH，形状均为 (脉冲数, 距离单元数)
# 用以下方式提取能量特征：
power_HH = np.mean(np.abs(echo_HH)**2, axis=0)
power_HV = np.mean(np.abs(echo_HV)**2, axis=0)
power_VH = np.mean(np.abs(echo_VH)**2, axis=0)
power_VV = np.mean(np.abs(echo_VV)**2, axis=0)



total_power = power_HH + power_HV + power_VH + power_VV + 1e-12  # 避免除以0


# 四个极化通道
echo_HH = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)
echo_HV = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)
echo_VH = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)
echo_VV = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)

# 功率计算（每个距离单元）
power_HH = np.mean(np.abs(echo_HH)**2, axis=0)
power_HV = np.mean(np.abs(echo_HV)**2, axis=0)
power_VH = np.mean(np.abs(echo_VH)**2, axis=0)
power_VV = np.mean(np.abs(echo_VV)**2, axis=0)

# 总功率
total_power = power_HH + power_HV + power_VH + power_VV + 1e-12

# 相对散射成分
surf_ratio = power_HH / total_power
body_ratio = (power_HV + power_VH) / (2 * total_power)
dihedral_ratio = power_VV / total_power

# 极化对称性（可选）
polarization_symmetry = np.abs(power_HV - power_VH) / (power_HV + power_VH + 1e-12)


print('polarization_symmetry Len:',len(polarization_symmetry))
print(polarization_symmetry)
