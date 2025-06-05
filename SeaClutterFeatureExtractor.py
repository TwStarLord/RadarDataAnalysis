import numpy as np
from scipy.fft import fft
from scipy.stats import entropy
from scipy.signal import spectrogram
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import polyfit
from skimage.measure import label, regionprops

class SeaClutterFeatureExtractor:
    def __init__(self, echo: np.ndarray):
        """
        echo: shape (num_pulse, num_range), complex I/Q data
        """
        self.echo = echo
        self.num_pulse, self.num_range = echo.shape

    def extract_hurst(self):
        def hurst(ts):
            N = len(ts)
            T = np.arange(1, N + 1)
            Y = np.cumsum(ts - np.mean(ts))
            R = np.max(Y) - np.min(Y)
            S = np.std(ts)
            return np.log(R / S) / np.log(N) if S != 0 else 0

        return np.array([hurst(np.abs(self.echo[:, i])) for i in range(self.num_range)])

    def extract_corr_dim(self, emb_dim=10, tau=1):
        def phase_space(ts):
            N = len(ts) - (emb_dim - 1) * tau
            return np.array([ts[i:i + emb_dim * tau:tau] for i in range(N)])

        def correlation_integral(X, r):
            from scipy.spatial.distance import pdist
            dists = pdist(X)
            return np.sum(dists < r) * 2.0 / (len(X) * (len(X) - 1))

        def corr_dim(ts):
            ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-12)
            X = phase_space(ts)
            r_vals = np.logspace(-2, 0, 20)
            C = [correlation_integral(X, r) for r in r_vals]
            C = np.array(C)
            valid = C > 0
            if np.sum(valid) < 5:
                return np.nan
            log_r = np.log(r_vals[valid])
            log_C = np.log(C[valid])
            slope, _ = polyfit(log_r, log_C, 1)
            return slope

        return np.array([corr_dim(np.abs(self.echo[:, i])) for i in range(self.num_range)])

    def extract_lyapunov(self, emb_dim=10, tau=1, max_t=20):
        def reconstruct(ts):
            N = len(ts) - (emb_dim - 1) * tau
            return np.array([ts[i:i + emb_dim * tau:tau] for i in range(N)])

        def estimate(ts):
            ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-12)
            X = reconstruct(ts)
            N = X.shape[0]
            tree = cKDTree(X)
            d_t = np.zeros(max_t)
            count = np.zeros(max_t)

            for i in range(N - max_t):
                _, idxs = tree.query(X[i], k=6)
                idxs = [j for j in idxs if abs(j - i) > emb_dim]
                for j in idxs:
                    for t in range(1, max_t):
                        if i + t < N and j + t < N:
                            dist = np.linalg.norm(X[i + t] - X[j + t])
                            d_t[t] += dist
                            count[t] += 1

            valid = count > 0
            t_vals = np.arange(max_t)[valid]
            log_d = np.log(d_t[valid] / count[valid] + 1e-12)
            if len(log_d) < 5:
                return np.nan
            model = LinearRegression().fit(t_vals.reshape(-1, 1), log_d)
            return model.coef_[0]

        return np.array([estimate(np.abs(self.echo[:, i])) for i in range(self.num_range)])

    def extract_time_features(self):
        amp = np.abs(self.echo)
        mean_amp = np.mean(amp, axis=0)
        rel_mean_amp = mean_amp / np.max(mean_amp)

        time_entropy = []
        scf = []

        for i in range(self.num_range):
            a = amp[:, i]
            hist, _ = np.histogram(a, bins=100, density=True)
            time_entropy.append(entropy(hist + 1e-12))
            scf.append(np.var(a) / (np.mean(a) ** 2 + 1e-12))

        return rel_mean_amp, np.array(time_entropy), np.array(scf)

    def extract_freq_features(self):
        doppler_peak = []
        doppler_entropy = []

        for i in range(self.num_range):
            spec = np.abs(fft(self.echo[:, i]))
            peak = np.max(spec)
            avg = np.mean(spec)
            doppler_peak.append(peak / (avg + 1e-12))

            s_half = spec[:self.num_pulse // 2]
            p = s_half / np.sum(s_half)
            doppler_entropy.append(entropy(p + 1e-12))

        return np.array(doppler_peak), np.array(doppler_entropy)

    def extract_time_freq_features(self):
        micro_doppler = []
        bright_area = []
        max_conn = []

        for i in range(self.num_range):
            f, t, Sxx = spectrogram(np.abs(self.echo[:, i]), fs=1000, nperseg=64)
            spec_sum = np.sum(Sxx, axis=1)
            micro_doppler.append(np.var(spec_sum))

            norm_S = Sxx / (np.max(Sxx) + 1e-12)
            binary_S = (norm_S > 0.5).astype(int)
            bright_area.append(np.sum(binary_S))

            regions = regionprops(label(binary_S))
            max_conn.append(max([r.area for r in regions]) if regions else 0)

        return np.array(micro_doppler), np.array(bright_area), np.array(max_conn)

    def extract_polarization_features(self, echo_HH, echo_HV, echo_VH, echo_VV):
        power_HH = np.mean(np.abs(echo_HH)**2, axis=0)
        power_HV = np.mean(np.abs(echo_HV)**2, axis=0)
        power_VH = np.mean(np.abs(echo_VH)**2, axis=0)
        power_VV = np.mean(np.abs(echo_VV)**2, axis=0)

        total = power_HH + power_HV + power_VH + power_VV + 1e-12

        surf = power_HH / total
        body = (power_HV + power_VH) / (2 * total)
        dihedral = power_VV / total
        symmetry = np.abs(power_HV - power_VH) / (power_HV + power_VH + 1e-12)

        return surf, body, dihedral, symmetry
