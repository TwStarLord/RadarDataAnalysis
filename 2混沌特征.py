import numpy as np
from scipy.spatial.distance import pdist

echo = np.random.randn(1000, 64) + 1j * np.random.randn(1000, 64)  # 示例数据
def phase_space_reconstruction(ts, emb_dim, tau):
    """相空间重构"""
    N = len(ts) - (emb_dim - 1) * tau
    return np.array([ts[i:i + emb_dim * tau:tau] for i in range(N)])

def correlation_integral(X, r):
    """计算相关积分 C(r)"""
    dists = pdist(X)
    return np.sum(dists < r) * 2.0 / (len(X) * (len(X) - 1))

# def compute_correlation_dimension(ts, emb_dim=5, tau=1, r_vals=None):
#     """Grassberger-Procaccia法估算关联维"""
#     ts = (ts - np.mean(ts)) / np.std(ts)
#     X = phase_space_reconstruction(ts, emb_dim, tau)
#     if r_vals is None:
#         r_vals = np.logspace(-2, 0, 20)
#     C = [correlation_integral(X, r) for r in r_vals]
#     log_C = np.log(C)
#     log_r = np.log(r_vals)
#     # 拟合线性区域
#     slope, _ = np.polyfit(log_r, log_C, 1)
#     return slope

# 关联维数（Correlation Dimension）
def compute_correlation_dimension(ts, emb_dim=5, tau=1, r_vals=None):
    ts = (ts - np.mean(ts)) / np.std(ts)
    X = phase_space_reconstruction(ts, emb_dim, tau)

    if r_vals is None:
        r_vals = np.logspace(-2, 0, 20)  # r ∈ [0.01, 1]
    C = [correlation_integral(X, r) for r in r_vals]

    # 仅保留 C > 0 的项，避免 log(0)
    C = np.array(C)
    valid = C > 0
    log_C = np.log(C[valid])
    log_r = np.log(r_vals)[valid]

    # import matplotlib.pyplot as plt
    # plt.plot(log_r, log_C, 'o-')
    # plt.xlabel('log(r)')
    # plt.ylabel('log(C(r))')
    # plt.title('Correlation Integral Curve')
    # plt.grid(True)
    # plt.show()

    # 拟合线性区域
    slope, _ = np.polyfit(log_r, log_C, 1)
    return slope



cor_dim_list = []

for i in range(echo.shape[1]):
    ts = np.abs(echo[:, i])
    d = compute_correlation_dimension(ts, emb_dim=5, tau=1)
    cor_dim_list.append(d)


print('cor_dim_list Len:',len(cor_dim_list))
print(cor_dim_list)

# from nolitsa import lyapunov
# import numpy as np
#
# def max_lyapunov(ts, dim=10, tau=1, window=10, maxt=30):
#     ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-12)
#     l_exp = lyapunov.mle(ts, dim, tau, window, maxt, metric='euclidean')
#     return np.mean(l_exp[1:])
#
#
#
# lyap_list = []
#
# for i in range(echo.shape[1]):
#     ts = np.abs(echo[:, i])
#     try:
#         l = max_lyapunov(ts,metric='euclidean')
#     except Exception as e:
#         print(f"距离单元 {i} 计算失败：{e}")
#         l = np.nan
#     lyap_list.append(l)
#
# print('lyap_list Len:',len(lyap_list))
# print(lyap_list)
# help(lyapunov.mle)


# ===
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

def phase_space_reconstruct(ts, emb_dim, tau):
    """相空间重构"""
    N = len(ts) - (emb_dim - 1) * tau
    return np.array([ts[i:i + emb_dim * tau:tau] for i in range(N)])

def kantz_lyapunov(ts, emb_dim=10, tau=1, max_t=20, min_neighbors=5, r_thresh=0.1):
    """
    基于 Kantz 方法估算最大 Lyapunov 指数
    参数：
        ts: 一维时间序列
        emb_dim: 嵌入维数
        tau: 延迟
        max_t: 最大时间步长
        min_neighbors: 最少邻居数量
        r_thresh: 邻域半径比例（相对于std）
    返回：
        最大 Lyapunov 指数估计值
    """
    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-12)
    X = phase_space_reconstruct(ts, emb_dim, tau)
    N = X.shape[0]

    # 建立 KD 树索引用于快速邻居查询
    tree = cKDTree(X)

    # 初始化平均距离序列
    d_t = np.zeros(max_t)
    count = np.zeros(max_t)

    for i in range(N - max_t):
        # 查找邻居（排除时间上太近的）
        _, idxs = tree.query(X[i], k=min_neighbors + 1, p=2)
        idxs = [j for j in idxs if abs(j - i) > emb_dim]

        for j in idxs:
            for t in range(1, max_t):
                if i + t < N and j + t < N:
                    dist = np.linalg.norm(X[i + t] - X[j + t])
                    d_t[t] += dist
                    count[t] += 1

    # 平均距离
    valid = count > 0
    t_vals = np.arange(max_t)[valid]
    log_d = np.log(d_t[valid] / count[valid] + 1e-12)

    # 线性拟合：log(d) = λ * t + c
    model = LinearRegression()
    model.fit(t_vals.reshape(-1, 1), log_d)
    lyap_exp = model.coef_[0]
    return lyap_exp

lyap_vals = []

for i in range(echo.shape[1]):
    ts = np.abs(echo[:, i])  # 使用幅度序列
    try:
        l = kantz_lyapunov(ts, emb_dim=8, tau=2, max_t=20)
    except Exception as e:
        print(f"距离单元 {i} 出错: {e}")
        l = np.nan
    lyap_vals.append(l)

print('lyap_vals Len:',len(lyap_vals))
print(lyap_vals)