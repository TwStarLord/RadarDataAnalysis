# 假设 echo 为复数 IQ 数据，shape = [pulse_num, range_num]
import numpy as np

from SeaClutterFeatureExtractor import SeaClutterFeatureExtractor

echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)

# 四个极化通道
echo_HH = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)
echo_HV = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)
echo_VH = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)
echo_VV = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)


extractor = SeaClutterFeatureExtractor(echo)

hurst = extractor.extract_hurst()
corr_dim = extractor.extract_corr_dim()
lyap = extractor.extract_lyapunov()
rel_amp, t_entropy, scf = extractor.extract_time_features()
doppler_peak, doppler_entropy = extractor.extract_freq_features()
micro_doppler, bright_area, max_conn = extractor.extract_time_freq_features()
surf, body, dihedral, symmetry = extractor.extract_polarization_features(echo_HH, echo_HV, echo_VH, echo_VV)

features = np.vstack([
    hurst, corr_dim, lyap,
    rel_amp, t_entropy, scf,
    doppler_peak, doppler_entropy,
    micro_doppler, bright_area, max_conn,
    surf, body, dihedral, symmetry
]).T  # shape = [num_range, num_features]


print(features.shape)
print(features)