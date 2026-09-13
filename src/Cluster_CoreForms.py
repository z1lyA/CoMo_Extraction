#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster_CoreForms.py —— 步骤⑤：出核心城市形态子类（UMAP + KMeans → T1..TK）

把 temp/2026-07-27/5building_level_coreform.py 重构成「读 config、出核心类型」的模块。

聚类流程（对齐参照研究 §2.3.2）：
  读步骤④产出的核心子空间（F_S 特征 + G_S 核心建筑），对核心建筑做 UMAP+KMeans，
  得到可命名的核心形态子类 T1..TK。依赖 ④ 的解释产物（fs_global.json / gnnexplainer_core.json）。

输出（results/Clustering/run_<ts>/）：
  - building_morph_types.csv        每栋建筑的特征 + 所属形态类型 Tx
  - K_Selection_Silhouette_Plot.png K 选择（轮廓系数）
  - umap_clusters.png               UMAP 聚类散点
  - clustering_metadata.json        最优 K / 特征 / 时间戳
  - representative_buildings.csv    每类代表建筑（用于科学命名）
  - function_morph_heatmap.png      功能×形态关联矩阵（仅模型路径有 class 语义时更准）
  - representative_buildings.shp    （extract_shp=true 时）代表性建筑空间文件
"""

import os
import sys

import sys as _sys
try:
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(errors='replace')
        _sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

# [FIX] Windows 上 UMAP 的 kNN 后端(pynndescent/numba)在多 OpenMP/numba 线程下会原生段错误，
# 表现为无 Python traceback、进程直接崩、退出码为怪异大数。限制线程数可规避。
# 必须在 import umap 之前设置，否则 numba 已在导入时锁定线程配置。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# [FIX] 允许 OpenMP(libiomp5md) 重复加载。运行环境同时含 numpy(MKL) 与 torch 两套 OpenMP 运行时，
# Windows 下易触发「DLL 初始化失败」(0xc06d007f)，表现为 sklearn KMeans 的 threadpoolctl 查询崩溃；
# 放开重复加载可规避该冲突，UMAP 子进程的原生段错误也可能一并缓解。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import pickle
import datetime
import yaml

import numpy as np
from . import numpy_pickle_shim  # 兼容 numpy2.x 保存的 pkl（numpy._core 别名到 core）
import multiprocessing as mp
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from umap import UMAP
except ImportError:
    print("[警告] 未安装 umap-learn，请先: pip install umap-learn")
    raise

try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

try:
    import joblib
except ImportError:
    joblib = None


# =============================================================================
# 特征中文名映射（出图 / 命名用）
# =============================================================================
FEATURE_NAME_CN = {
    "Area": "面积", "Perimeter": "周长", "Longest_chord_length": "最长弦长",
    "Mean_radius": "平均半径", "Covered_area_ratio": "覆盖率", "Height": "高度",
    "SBRO": "短边相对方向", "LCO": "最长弦方向", "Bisector_orientation": "平分线方向",
    "Weighted_orientation": "加权方向", "Complexity_index": "复杂度指数", "IPQ": "等周商",
    "Fractality": "分形维数", "Max_circularity": "最大圆度", "Gibbs_compactness": "吉布斯紧凑度",
    "Elongation_index": "伸长指数", "Ellipticity": "椭圆度", "Solidity": "凸度",
    "DCM": "数字紧凑度", "Exchange_index": "交换指数",     "BCI": "博伊斯-克拉克形状指数",
    "Cross_moment_mu11": "交叉矩",
    "Variance_spreading": "铺展方差",
}


def name_clusters(type_zmean, top_n=2):
    """根据每类特征 z-score 均值签名生成描述性名字（中英都存）。
    取 |z均值| 最大的 top_n 个特征，按方向 High(>0)/Low(<0) 拼接。
    返回 {morph_type: {'english','chinese','signature':[{feature,cn,zmean,direction}]}}。"""
    names = {}
    for mt, zmap in type_zmean.items():
        ranked = sorted(zmap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        sig, en_parts, cn_parts = [], [], []
        for f, z in ranked:
            direction = "High" if z >= 0 else "Low"
            cn_dir = "高" if z >= 0 else "低"
            cn = FEATURE_NAME_CN.get(f, f)
            sig.append({"feature": f, "cn": cn, "zmean": float(z), "direction": direction})
            en_parts.append(f"{direction}-{f}")
            cn_parts.append(f"{cn_dir}{cn}")
        names[mt] = {
            "english": f"T{mt}: " + "/".join(en_parts),
            "chinese": f"T{mt}: " + "/".join(cn_parts),
            "signature": sig,
        }
    return names


def plot_cluster_profiles(df, feature_names, type_zmean, type_names, output_dir):
    """每类特征画像（z-score 均值热力图）+ 各类建筑数柱状图。英文标签。"""
    mts = sorted(type_zmean.keys())
    sizes = [int((df['morph_type'] == mt).sum()) for mt in mts]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(mts) * 0.9)))
    axes[0].bar([type_names[m]['english'] for m in mts], sizes, color="#378ADD")
    axes[0].set_title("Cluster sizes (building counts)"); axes[0].set_ylabel("count")
    axes[0].tick_params(axis='x', rotation=30, labelsize=8)
    mat = np.array([[type_zmean[m][f] for f in feature_names] for m in mts])
    sns.heatmap(mat, xticklabels=feature_names,
                yticklabels=[type_names[m]['english'] for m in mts],
                cmap="RdBu_r", center=0, ax=axes[1], annot=False)
    axes[1].set_title("Per-cluster feature z-score mean (signature)")
    axes[1].tick_params(axis='x', rotation=60, labelsize=7)
    fig.tight_layout()
    p = os.path.join(output_dir, "cluster_profiles.png")
    fig.savefig(p, dpi=300); plt.close(fig)


def plot_cluster_radar(df, feature_names, output_dir):
    """各形态类型 T 的特征维度多边形图（雷达/蜘蛛网图）。
    数值 = 该特征在全量 G_S 建筑上的 min-max 归一化 [0,1]（0=全局最小, 1=全局最大），
    使不同量纲特征可在同一多边形上比较。方向类特征为线性归一，仅作相对指示，
    真实朝向须结合 representative_buildings.csv 目视解译。
    产出：① 总览图 cluster_profiles_radar.png（所有类型叠在一张，便于横向比较）
          ② 每类单独图 cluster_profiles_radar_T{k}.png（同目录，便于逐类细看/命名）"""
    mts = sorted(df['morph_type'].unique())
    gmin = df[feature_names].min()
    gmax = df[feature_names].max()
    rng = (gmax - gmin).replace(0.0, 1e-12)
    norm = (df[feature_names] - gmin) / rng
    means = norm.groupby(df['morph_type']).mean()

    N = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    cmap = plt.get_cmap('tab10')

    # ---- ① 总览：所有类型叠在一张 ----
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, mt in enumerate(mts):
        vals = means.loc[mt].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=cmap(i % 10), linewidth=1.8, label=f"T{int(mt)}")
        ax.fill(angles, vals, color=cmap(i % 10), alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Per-cluster feature profile (min-max normalized 0-1; direction features linear-only)",
                 fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.10), fontsize=8)
    fig.tight_layout()
    p = os.path.join(output_dir, "cluster_profiles_radar.png")
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)

    # ---- ② 每类单独一张 ----
    per_class_files = []
    for i, mt in enumerate(mts):
        vals = means.loc[mt].tolist()
        vals += vals[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, vals, color=cmap(i % 10), linewidth=2.0)
        ax.fill(angles, vals, color=cmap(i % 10), alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"T{int(mt)} feature profile (min-max normalized 0-1)", fontsize=11)
        fig.tight_layout()
        pt = os.path.join(output_dir, f"cluster_profiles_radar_T{int(mt)}.png")
        fig.savefig(pt, dpi=300, bbox_inches='tight'); plt.close(fig)
        per_class_files.append(pt)
    return per_class_files


def save_cluster_feature_means(df, feature_names, output_dir):
    """每类 × 每特征 z-score 均值（已标准化，非原始量纲；仅用于相对高低比较，辅助目视命名）。"""
    means = df.groupby('morph_type')[feature_names].mean()
    means.index = [f"T{int(m)}" for m in means.index]
    means.index.name = "morph_type"
    p = os.path.join(output_dir, "cluster_feature_means.csv")
    means.to_csv(p, encoding="utf-8-sig")
    return p


def plot_naming_summary(type_names, output_dir):
    """自动命名提示汇总表图（英文，避免中文乱码；仅供目视解译参考，非最终命名）。"""
    mts = sorted(type_names.keys())
    fig, ax = plt.subplots(figsize=(11, max(3, len(mts) * 0.8)))
    ax.axis('off')
    cols = ["Type", "English name", "Top features (feature=z-mean,dir)"]
    rows = []
    for m in mts:
        sig = type_names[m]['signature']
        sigtxt = "; ".join(f"{s['feature']}={s['zmean']:.2f}({s['direction']})" for s in sig)
        rows.append([f"T{m}", type_names[m]['english'], sigtxt])
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='left')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title("Auto-generated naming hints (replace with your visual-interpretation names)")
    fig.tight_layout()
    p = os.path.join(output_dir, "cluster_naming_summary.png")
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)


# =============================================================================
# 工具
# =============================================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_latest_fs_global(run_dir=None):
    """返回 ④ 产出的 fs_global.json（含 selected_features 全局 F_S 选中特征）。
    优先在当前 run 目录的 03_Explainer 下找，回退旧扁平 results/Explainer_Importance。"""
    cands = []
    if run_dir:
        d = os.path.join(run_dir, "03_Explainer")
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f == "fs_global.json":
                        cands.append(os.path.join(root, f))
    base = "results/Explainer_Importance"
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for f in files:
                if f == "fs_global.json":
                    cands.append(os.path.join(root, f))
    if not cands:
        return None
    latest = max(cands, key=os.path.getmtime)
    try:
        with open(latest, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def find_latest_gnn_explainer_core(run_dir=None):
    """返回 ④ 产出的 gnnexplainer_core.json（含 G_S 核心建筑 ID）路径。
    优先当前 run 目录的 03_Explainer，回退旧扁平 results/Explainer_Importance。找不到返回 None。"""
    cands = []
    if run_dir:
        d = os.path.join(run_dir, "03_Explainer")
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f == "gnnexplainer_core.json":
                        cands.append(os.path.join(root, f))
    base = "results/Explainer_Importance"
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for f in files:
                if f == "gnnexplainer_core.json":
                    cands.append(os.path.join(root, f))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def load_gs_building_ids(core_path):
    """从 gnnexplainer_core.json 收集所有 G_S 核心建筑 ID（按 '|' 拆分归一，兼容合并建筑）。"""
    with open(core_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    gs_set = set()
    for b in data.get("blocks", []):
        for bid in b.get("G_S_building_ids", []):
            for part in str(bid).split("|"):
                if part:
                    gs_set.add(part)
    return gs_set


def parse_primary_fid(token):
    s = str(token).strip()
    parts = [p.strip() for p in s.split('|') if str(p).strip() != '']
    if not parts:
        return None
    last = parts[-1]
    return int(last) if last.isdigit() else last


# =============================================================================
# UMAP 封装（进程内执行；OpenMP/MKL 已修复，不再需要子进程隔离，见 run_umap_safe 内注释）
# =============================================================================
def run_umap_safe(feat_mat, n_neighbors, min_dist, random_state, timeout=900):
    """进程内运行 UMAP 降维，返回 (embedding, reducer, None) 成功 / (None, None, error_msg) 失败。

    历史：曾用 spawn 子进程隔离 UMAP，以规避 Windows 上损坏 OpenMP 运行时导致的
    原生段错误。2026-08-01 修复 OpenMP/MKL 后，进程内 UMAP 已稳定（用户实测
    umap.UMAP(n_jobs=1).fit_transform 正常返回）；而子进程方案在 Windows 上会重导
    torch_geometric 等重型模块、且对 19001/1919 点均卡满 900s 超时，故改为进程内
    执行 + try/except，失败时由调用方回退 PCA。
    reducer 一并返回，供下游（步骤⑥）将同一降维器套用到全部建筑。
    """
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                       n_components=2, random_state=random_state, n_jobs=1)
        emb = reducer.fit_transform(feat_mat)
        return emb, reducer, None
    except Exception as e:
        return None, None, "UMAP 进程内执行失败: %s: %s" % (type(e).__name__, e)


# =============================================================================
# 聚类主流程
# =============================================================================
# =============================================================================
# 纯 numpy 聚类（规避 sklearn 的 threadpoolctl 在 Windows 损坏 OpenMP 运行时下的崩溃）
# =============================================================================
def _kmeans_plusplus_init(X, k, rng):
    """k-means++ 初始化中心，返回 (k, d) 数组。"""
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / (d2.sum() + 1e-12)
        nxt = rng.choice(n, p=probs)
        centers.append(X[nxt])
        nd = ((X - X[nxt]) ** 2).sum(1)
        d2 = np.minimum(d2, nd)
    return np.stack(centers)


def kmeans_numpy(X, k, random_state=42, n_init=10, max_iter=300, tol=1e-4):
    """纯 numpy Lloyd's KMeans（k-means++ 初始化 + 多次重启取最优 inertia）。
    完全不依赖 sklearn，规避运行环境 threadpoolctl/OpenMP 崩溃。"""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    x_sq = (X ** 2).sum(1)[:, None]   # 展开距离公式，避免 (n, k, d) 中间张量
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = _kmeans_plusplus_init(X, k, rng)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = x_sq + (centers ** 2).sum(1)[None, :] - 2.0 * (X @ centers.T)
            new_labels = dists.argmin(1)
            new_centers = np.empty_like(centers)
            for j in range(k):
                m = new_labels == j
                new_centers[j] = X[m].mean(0) if m.any() else centers[j]
            shift = ((new_centers - centers) ** 2).sum()
            centers, labels = new_centers, new_labels
            if shift <= tol:
                break
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def zscore_matrix(X):
    """列向 z-score 标准化。供「在原始特征空间评估轮廓系数」时消除量纲差异。"""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def largest_cluster_frac(labels):
    """最大簇占比。用于识别「一个簇吞掉绝大多数样本」的退化解。"""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 1.0
    _, cnt = np.unique(labels, return_counts=True)
    return float(cnt.max() / labels.size)


def silhouette_numpy(X, labels, max_samples=20000, random_state=42):
    """纯 numpy 轮廓系数（分块计算，避免 N×N 全矩阵占用过大内存）。
    不依赖 sklearn，规避 threadpoolctl 崩溃。

    [FIX 2026-08-02] 修正 a（簇内平均距离）的重大计算错误：
      旧写法先把「到各簇的距离和」除以 size 得 mean_dist，之后又除 (size-1)，
      等价于 sum/(size*(size-1)) —— 多除了一次簇大小，a 被缩小约 size 倍，
      于是 s=(b-a)/max(a,b) → b/b → 1。簇越大虚高越狠：
        6 点分 2 簇 -> 0.9552（正确 0.8657）
        1919 点分 3 簇（每簇约 640）-> 直冲 0.999
      这正是「自动选 K 永远选最小 K、轮廓系数恒 0.999x」的根因，
      与 UMAP min_dist 无关（min_dist 只是雪上加霜）。
      现改回标准定义：a = 簇内距离和/(size-1)，b = 到最近其他簇的平均距离。

    max_samples: 点数超过该值时随机采样后再算，避免 O(n²) 距离矩阵在全量广州数据
      （几十万建筑）上内存/耗时爆炸。采样只影响评分精度，不影响聚类结果本身。
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, labels = X[idx], labels[idx]
    n = X.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    k = len(uniq)
    lab_idx = np.searchsorted(uniq, labels)      # 归一到 0..k-1，避免空簇导致索引错位
    C = np.zeros((n, k))
    C[np.arange(n), lab_idx] = 1.0
    sizes = C.sum(0)
    sum_dist = np.zeros((n, k))                  # [FIX] 保留“距离和”，不在此处除 size
    # 距离用 |x|²+|y|²-2x·y 展开分块算：中间张量从 (chunk, n, d) 降到 (chunk, n)，
    # 20000 点时内存从 GB 级降到 ~80MB，全量广州数据也扛得住。
    X32 = X.astype(np.float32)
    C32 = C.astype(np.float32)
    sq = (X32 ** 2).sum(1)
    chunk = 1000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (X32[s:e] @ X32.T)
        np.maximum(d2, 0.0, out=d2)
        sum_dist[s:e] = np.sqrt(d2) @ C32
    a = sum_dist[np.arange(n), lab_idx] / np.maximum(sizes[lab_idx] - 1, 1)
    mean_other = sum_dist / np.maximum(sizes, 1)
    mean_other[np.arange(n), lab_idx] = np.inf
    b = mean_other.min(1)
    s = (b - a) / np.maximum(a, b)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(s))


def run_clustering(rows, feature_names, output_dir, cfg, all_feature_names=None):
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("没有有效的建筑特征数据可供聚类。")
    feat_mat = df[feature_names].values.astype(float)
    # [FIX] 双保险：即便上游已 nan_to_num，这里再核一次，避免非有限值触发 UMAP 原生崩溃
    if not np.isfinite(feat_mat).all():
        n_bad = int(np.sum(~np.isfinite(feat_mat)))
        col_med = np.nanmedian(feat_mat, axis=0)
        feat_mat = np.where(np.isfinite(feat_mat), feat_mat, col_med)
        print(f"    [警告] feat_mat 含 {n_bad} 个非有限值(NaN/inf)，已用列中位数填充")

    cl = cfg['clustering']
    n_neigh = int(cl['umap_n_neighbors']); min_dist = float(cl['umap_min_dist'])
    min_k = int(cl['min_k']); max_k = int(cl['max_k']); rs = int(cl['random_state'])

    dim_red = str(cl.get('dim_reduction', 'auto')).lower()
    print(f"    使用 {len(feature_names)} 个特征做降维 (n_neighbors={n_neigh}, min_dist={min_dist})...")

    embedding = None
    dimred_model = None          # 拟合好的降维器（UMAP reducer 或 PCA），供步骤⑥套全部建筑
    dimred_kind = None
    if dim_red == 'pca':
        print("    [降维] 直接走 PCA（config: dim_reduction=pca）")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=rs)
        embedding = pca.fit_transform(feat_mat)
        dimred_model, dimred_kind = pca, 'pca'
    else:
        # auto（默认）或 umap：进程内跑 UMAP（OpenMP 已修复，不再需要子进程隔离）。
        # 异常由 run_umap_safe 内部 try/except 捕获；失败时调用方回退 PCA，保证聚类不中断。
        print("    [降维] UMAP 进程内运行（n_jobs=1），请稍候...")
        emb, reducer, err = run_umap_safe(feat_mat, n_neigh, min_dist, rs, timeout=900)
        if emb is not None:
            embedding = emb
            dimred_model, dimred_kind = reducer, 'umap'
        elif dim_red == 'umap':
            raise RuntimeError(
                f"UMAP 子进程失败({err})。Windows 上 UMAP 的 numba/pynndescent 后端易原生崩溃；"
                f"请升级 numba 后重试，或把 clustering.dim_reduction 改为 auto/pca 以回退 PCA。")
        else:  # auto 回退 PCA
            print(f"    [警告] {err}，回退用 PCA 做 2D 嵌入（聚类仍可进行，方法等价）")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=rs)
            embedding = pca.fit_transform(feat_mat)
            dimred_model, dimred_kind = pca, 'pca'
    df['embed_x'], df['embed_y'] = embedding[:, 0], embedding[:, 1]

    # ---------------------------------------------------------------
    # 自动选 K：【评估空间与降维空间解耦】
    #   聚类仍在 UMAP 嵌入空间进行（方法链与参照研究一致），
    #   但轮廓系数一律回到「原始 F_S 标准化特征空间」计算。
    #   原因：umap_min_dist 的作用就是控制点在嵌入空间挤得多紧——
    #   在嵌入空间打分等于「先按参数把尺子拉长，再问哪个参数量得最高」，
    #   必然偏向极小 min_dist 与极小 K。原始空间才是所有超参共用的固定标尺。
    # ---------------------------------------------------------------
    feat_eval = zscore_matrix(feat_mat)

    # 大数据保护：K 搜索在随机子样本上做（最终聚类仍在全量嵌入上跑），
    # 否则全量广州几十万核心建筑 × (max_k-min_k+1) 次 KMeans 会跑到天荒地老。
    k_search_max = int(cl.get('k_search_max_samples', 20000))
    if embedding.shape[0] > k_search_max:
        _rng = np.random.default_rng(rs)
        _sub = _rng.choice(embedding.shape[0], size=k_search_max, replace=False)
        emb_s, eval_s = embedding[_sub], feat_eval[_sub]
        print(f"    [K搜索] 样本 {embedding.shape[0]} > {k_search_max}，"
              f"K 搜索在随机子样本上进行（最终聚类仍用全量）")
    else:
        emb_s, eval_s = embedding, feat_eval

    best_k, best_score, scores = min_k, -1, []
    for k in range(min_k, max_k + 1):
        labels = kmeans_numpy(emb_s, k, random_state=rs, n_init=10)
        sc = silhouette_numpy(eval_s, labels, random_state=rs)     # 主判据：原始空间
        sc_emb = silhouette_numpy(emb_s, labels, random_state=rs)  # 仅供参考：嵌入空间
        scores.append({'k': k, 'silhouette_score': sc,
                       'silhouette_embedding': sc_emb,
                       'largest_cluster_frac': largest_cluster_frac(labels)})
        if sc > best_score:
            best_score, best_k = sc, k

    pd.DataFrame(scores).to_csv(os.path.join(output_dir, "K_Selection_Silhouette_Plot.csv"), index=False)
    plt.figure(figsize=(8, 5))
    ks = [s['k'] for s in scores]
    ss = [s['silhouette_score'] for s in scores]
    ss_emb = [s['silhouette_embedding'] for s in scores]
    plt.plot(ks, ss, marker='o', linestyle='-', color='b', label='Raw F_S space (used for selection)')
    plt.plot(ks, ss_emb, marker='.', linestyle=':', color='gray', alpha=0.7,
             label='UMAP embedding space (reference only)')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    plt.xlabel('Number of Clusters (K)'); plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K Selection'); plt.legend(); plt.grid(True, alpha=0.6)
    plt.savefig(os.path.join(output_dir, "K_Selection_Silhouette_Plot.png"), dpi=300); plt.close()
    print(f"    自动选择最优聚类数 K = {best_k} "
          f"(原始F_S空间轮廓系数 {best_score:.4f}；嵌入空间 {scores[best_k - min_k]['silhouette_embedding']:.4f} 仅参考)")

    final_labels = kmeans_numpy(embedding, best_k, random_state=rs, n_init=10)
    df['morph_type'] = final_labels + 1

    # 每类特征 z-score 均值签名（供画像/雷达图使用）
    _feat = df[feature_names].astype(float)
    _fmean, _fstd = _feat.mean(), _feat.std().replace(0.0, 1e-12)
    feat_z = (_feat - _fmean) / _fstd
    type_zmean = {}
    for mt in sorted(df['morph_type'].unique()):
        sub = feat_z[df['morph_type'] == mt]
        type_zmean[int(mt)] = {f: float(sub[f].mean()) for f in feature_names}

    # 自动命名：auto_name=false（默认）时只保留 T1..TK 机械编号，命名交给用户目视解译；
    # 无论是否启用，都额外产出 cluster_name_suggestions.json 作为提示（不强制）。
    auto_name = bool(cl.get('auto_name', False))
    suggestions = name_clusters(type_zmean, top_n=int(cl.get('name_top_n', 2)))
    if auto_name:
        type_names = suggestions
    else:
        type_names = {mt: {'english': f"T{mt}", 'chinese': f"T{mt}", 'signature': []}
                      for mt in type_zmean}
    suggestions_path = os.path.join(output_dir, "cluster_name_suggestions.json")
    with open(suggestions_path, 'w', encoding='utf-8') as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)

    df.to_csv(os.path.join(output_dir, "building_morph_types.csv"), index=False, encoding="utf-8-sig")

    # 每类特征画像（热力图）+ 多边形雷达图 + 原始均值表
    try:
        plot_cluster_profiles(df, feature_names, type_zmean, type_names, output_dir)
        print(f"    每类特征画像(热力图)已保存: {os.path.join(output_dir, 'cluster_profiles.png')}")
    except Exception as e:
        print(f"    [warn] 每类特征画像生成失败: {e}")
    try:
        per_cls = plot_cluster_radar(df, feature_names, output_dir)
        print(f"    每类特征多边形雷达图(总览)已保存: {os.path.join(output_dir, 'cluster_profiles_radar.png')}")
        print(f"    各类单独雷达图已保存: {len(per_cls)} 张 -> cluster_profiles_radar_T{{1..K}}.png")
    except Exception as e:
        print(f"    [warn] 雷达图生成失败: {e}")
    try:
        mp = save_cluster_feature_means(df, feature_names, output_dir)
        print(f"    每类特征 z-score 均值表已保存(已标准化, 非原始量纲): {mp}")
    except Exception as e:
        print(f"    [warn] 特征均值表生成失败: {e}")

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('Set1', best_k)
    for i, mt in enumerate(sorted(df['morph_type'].unique())):
        mask = df['morph_type'] == mt
        plt.scatter(df.loc[mask, 'embed_x'], df.loc[mask, 'embed_y'],
                    color=cmap(i), s=5, alpha=0.9,
                    label=type_names.get(int(mt), {}).get('english', f'Type {mt}'))
    plt.title(f"Core Urban Morphological Clusters (K={best_k})"); plt.xlabel("Dim-1"); plt.ylabel("Dim-2")
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1)); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"), dpi=300, bbox_inches='tight'); plt.close()

    # 代表建筑（每类离簇心最近的代表）
    n_rep = int(cl.get('num_representative', 30))
    reps = []
    for mt in sorted(df['morph_type'].unique()):
        sub = df[df['morph_type'] == mt]
        cx, cy = sub['embed_x'].mean(), sub['embed_y'].mean()
        d = np.sqrt((sub['embed_x'] - cx) ** 2 + (sub['embed_y'] - cy) ** 2)
        for rank, idx in enumerate(d.nsmallest(min(n_rep, len(d))).index, 1):
            r = sub.loc[idx]
            reps.append({'morph_type': int(mt), 'rank': rank,
                         'building_ids_str': r.get('building_ids_str', ''),
                         'block_id': r.get('block_id', 'N/A'),
                         'class_id': r.get('class_id', 'N/A'),
                         'dist_to_centroid': float(d[idx])})
    rep_df = pd.DataFrame(reps)
    rep_df.to_csv(os.path.join(output_dir, "representative_buildings.csv"), index=False, encoding="utf-8-sig")

    # 功能×形态关联（若有 class 信息）
    if 'class_id' in df.columns:
        try:
            names = cfg.get('_class_names', [f'class_{i}' for i in range(10)])
            tmp = df.copy()
            tmp['class_name'] = tmp['class_id'].apply(lambda c: names[int(c)] if int(c) < len(names) else str(c))
            mat = pd.crosstab(tmp['class_name'], tmp['morph_type'], normalize='index')
            plt.figure(figsize=(12, 8))
            sns.heatmap(mat, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Urban Function vs Morphological Type Matrix")
            plt.savefig(os.path.join(output_dir, "function_morph_heatmap.png"), dpi=300); plt.close()
        except Exception as e:
            print(f"    [warn] 功能×形态热力图生成失败: {e}")

    # 元数据 + 模型
    metadata = {'feature_names': feature_names, 'best_k': best_k, 'num_clusters': best_k,
                'cluster_labels': list(range(1, best_k + 1)),
                'cluster_names': type_names,
                'cluster_names_manual': None,
                'cluster_name_suggestions': 'cluster_name_suggestions.json',
                'timestamp': datetime.datetime.now().isoformat()}
    with open(os.path.join(output_dir, "clustering_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    # 保存最终簇心（kmeans 已改为纯 numpy 实现，故此处保存簇心供下游/复现使用，不再依赖 joblib 序列化未定义对象）
    centroids = []
    for mt in sorted(df['morph_type'].unique()):
        mask = df['morph_type'] == mt
        centroids.append([float(df.loc[mask, 'embed_x'].mean()), float(df.loc[mask, 'embed_y'].mean())])
    np.save(os.path.join(output_dir, "cluster_centroids.npy"), np.array(centroids, dtype=float))
    # 保存拟合好的降维器（UMAP reducer 或 PCA）+ 元信息，供步骤⑥把同一降维器套到「全部建筑」
    # （论文方法：用已保存的 UMAP/KMeans 对每地块内全部建筑分类）。joblib 不可用时跳过并告警。
    if joblib is not None and dimred_model is not None:
        joblib.dump(dimred_model, os.path.join(output_dir, "dimred_model.joblib"))
        with open(os.path.join(output_dir, "dimred_info.json"), 'w') as f:
            json.dump({'kind': dimred_kind, 'feature_names': feature_names,
                       'best_k': best_k, 'n_features': len(feature_names)}, f, indent=2)
        print(f"    降维模型已保存: dimred_model.joblib (kind={dimred_model})")
    else:
        print("    [warn] joblib 不可用或降维模型为空，未保存 dimred_model.joblib（步骤⑥全量分类将不可用）")
    print(f"    产出已保存: building_morph_types.csv / umap_clusters.png / clustering_metadata.json (K={best_k}) / cluster_centroids.npy")
    return df, best_k, rep_df, type_names, suggestions


def extract_representative_shp(df_all, rep_df, cfg, output_dir):
    if not cfg['clustering'].get('extract_shp', False) or not HAS_GEO:
        print(">>> 跳过 shp 提取（未启用或 geopandas 不可用）")
        return
    form_shp = cfg['data'].get('buildings_shp')
    func_shp = cfg['data'].get('landuse_shp')
    if not form_shp or not func_shp:
        print(">>> 跳过 shp 提取（未配置 shp 路径）")
        return
    print(">>> 导出代表性建筑 shp...")
    try:
        bgdf = gpd.read_file(form_shp)
        bid_col = next((c for c in ['buildingid', 'building_id', 'fid', 'FID', 'id', 'ID'] if c in bgdf.columns), None)
        if bid_col is None:
            print("    警告: 找不到建筑 ID 字段"); return
        out_rows = []
        for _, r in rep_df.iterrows():
            fid = parse_primary_fid(r['building_ids_str'])
            if fid is None:
                continue
            row = bgdf[bgdf[bid_col] == fid]
            if not row.empty:
                nr = row.iloc[0].copy()
                nr['morph_type'] = r['morph_type']; nr['rank'] = r['rank']; nr['class_id'] = r['class_id']
                out_rows.append(nr)
        if out_rows:
            gdf = gpd.GeoDataFrame(out_rows, crs=bgdf.crs)
            out_shp = os.path.join(output_dir, "representative_buildings.shp")
            gdf.to_file(out_shp)
            print(f"    代表性建筑 shp 已保存: {out_shp} ({len(gdf)} 个)")
    except Exception as e:
        print(f"    错误: shp 提取失败: {e}")


def extract_core_features(df, feature_names, best_k, cfg, output_dir):
    """
    聚类后自动提取「核心城市形态特征集」——即用户所述流程：
    对每个形态类型取 top-N 最显著特征，把各类型的 top 放一起、剔除重复，
    得到精炼特征列表，供下游「关键子图/解释」作为输入特征维度。

    算法：
      1. 对全部 feature_names 做全局 z-score 标准化（消除量纲差异）
      2. 对每个 morph_type 计算类内标准化均值（该类型相对全局的偏移）
      3. 取该类型 |偏移| 最大的 top-N 个特征作为其「特征签名」
      4. 合并所有类型的 top 特征、按出现顺序去重 -> core_features
    """
    cl = cfg['clustering']
    top_n = int(cl.get('top_n_per_type', 4))

    feats = df[feature_names].astype(float)
    g_mean = feats.mean()
    g_std = feats.std().replace(0.0, 1e-12)
    z = (feats - g_mean) / g_std

    per_type_top = {}
    core_set = []
    for mt in sorted(df['morph_type'].unique()):
        sub_z = z[df['morph_type'] == mt]
        type_mean = sub_z.mean()
        ranked = type_mean.abs().sort_values(ascending=False)
        top_feats = [str(x) for x in ranked.head(top_n).index]
        per_type_top[int(mt)] = top_feats
        for f in top_feats:
            if f not in core_set:
                core_set.append(f)

    print(f"    每类 top-{top_n} 特征合并去重 -> 核心城市形态特征集 ({len(core_set)} 个): {core_set}")

    out = {
        'core_features': core_set,
        'per_type_top_features': {str(k): v for k, v in per_type_top.items()},
        'top_n_per_type': top_n,
        'method': 'per-type z-score mean abs top-N, merged & deduped',
        'num_clusters': int(best_k),
        'source_feature_pool': list(feature_names),
    }
    jpath = os.path.join(output_dir, 'core_morph_features.json')
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    tpath = os.path.join(output_dir, 'core_morph_features.txt')
    with open(tpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(core_set))
    print(f"    核心特征列表已保存: {jpath}")
    return out


# =============================================================================
# 主程序
# =============================================================================
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    if not os.path.exists(config_path):
        print(f"错误: 未找到配置文件 {config_path}")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cl = cfg['clustering']
    _run_dir = os.environ.get('PIPELINE_RUN_DIR')
    graph_dir = cfg['training_paths']['graph_data_dir']
    gd_path = os.path.join(graph_dir, "graph_dataset.pt")
    meta_path = os.path.join(graph_dir, "metadata.pkl")
    if not os.path.exists(gd_path) or not os.path.exists(meta_path):
        print(f"错误: 图数据不存在: {gd_path}")
        return

    print(f"=== 步骤⑤ 出核心形态子类 | 图数据: {graph_dir} ===")
    graphs = torch.load(gd_path, map_location='cpu', weights_only=False)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    all_feature_names = meta.get('feature_names', [f'feat_{i}' for i in range(graphs[0].x.shape[1])])
    class_names = meta.get('class_names', [f'class_{i}' for i in range(10)])
    cfg['_class_names'] = class_names

    # 特征选择：决定聚类用的「特征子空间」
    #   feature_source = fs_from_explainer（默认，对齐参照研究）：读 ④ 全局 F_S 的 selected_features
    #   feature_source = custom（真正无 GNN，④ 可跳过）：用 config.custom_features
    #   feature_source = all：全部 22 维
    feature_source = str(cl.get('feature_source', 'fs_from_explainer')).lower()
    if feature_source == 'fs_from_explainer':
        fs_global = find_latest_fs_global(_run_dir)
        if fs_global is None:
            print("错误: clustering.feature_source=fs_from_explainer，但找不到步骤④的 fs_global.json。")
            print("      请先运行步骤④（解释/XGNN），或把 feature_source 改为 'custom'（无 GNN 模式）。")
            return
        sel = fs_global.get('selected_features', [])
        feature_names = [f for f in sel if f in all_feature_names]
        if not feature_names:
            print("错误: fs_global.json 的 selected_features 为空或与图特征不匹配，无法聚类。")
            return
        print(f"    聚类特征来自 ④ 全局 F_S（核心城市形态特征集）: {len(feature_names)} 个 -> {feature_names}")
    elif feature_source == 'all':
        feature_names = list(all_feature_names)
        print(f"    聚类特征: 全部 {len(feature_names)}/{len(all_feature_names)} 个")
    else:  # custom（维持原 config.custom_features）
        feature_names = [f for f in cl.get('custom_features', []) if f in all_feature_names]
        if not feature_names:
            feature_names = list(all_feature_names)
        print(f"    聚类特征（custom）: {len(feature_names)}/{len(all_feature_names)} 个 -> {feature_names}")

    # G_S 核心建筑筛选：仅聚类 ④ 解释出的「对分类有贡献的建筑」，大幅减少 UMAP 点数
    # 对应论文 §2.3.2 Step1 用 GNNExplainer 提取的核心子图建筑(G_S)做形态聚类。
    use_gs = bool(cl.get('use_gs_buildings', True))
    gs_set = None
    if use_gs:
        core_path = find_latest_gnn_explainer_core(_run_dir)
        if core_path:
            gs_set = load_gs_building_ids(core_path)
            print(f"    [G_S 筛选] 读取核心建筑 {len(gs_set)} 个（④ gnnexplainer_core.json），仅对这些建筑做 UMAP+KMeans")
        else:
            print("    [G_S 筛选] 未找到 gnnexplainer_core.json（步骤④未跑？），回退聚类全部建筑")

    # 收集建筑记录：仅保留 ④ 提取的 G_S 核心建筑（use_gs_buildings=true 时）
    print(">>> 对核心建筑形态特征做聚类（读 ④ 的 G_S 核心建筑 + F_S 特征）")
    rows = []
    for gi, g in enumerate(graphs):
        x = g.x.cpu().numpy()
        bstrs = getattr(g, 'building_ids_str', [''] * x.shape[0])
        yval = int(g.y.item()) if hasattr(g.y, 'item') else int(g.y)
        bid = getattr(g, 'block_id', -1)
        for ni in range(x.shape[0]):
            # [G_S 筛选] 若启用且找到 G_S，跳过非核心建筑（大幅减少 UMAP 点数）
            bid_str = bstrs[ni] if ni < len(bstrs) else ''
            if gs_set is not None and not any(p in gs_set for p in str(bid_str).split('|')):
                continue
            row = {'graph_index': gi, 'node_index': ni, 'block_id': bid,
                   'class_id': yval, 'building_ids_str': bid_str}
            for fname in feature_names:
                row[fname] = float(x[ni, all_feature_names.index(fname)])
            rows.append(row)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _run_dir = os.environ.get('PIPELINE_RUN_DIR')
    out_dir = os.path.join(_run_dir, '04_Clustering', f"run_{ts}") if _run_dir else os.path.join("results/Clustering", f"run_{ts}")
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "pipeline_config.json"), 'w', encoding='utf-8') as f:
        json.dump({'timestamp': ts, 'clustering': cl, 'feature_names': feature_names}, f, indent=4, ensure_ascii=False)

    df, best_k, rep_df, type_names, suggestions = run_clustering(rows, feature_names, out_dir, cfg)
    extract_representative_shp(df, rep_df, cfg, out_dir)

    final_dir = f"{out_dir}_K{best_k}"
    try:
        os.rename(out_dir, final_dir); out_dir = final_dir
    except Exception:
        pass

    # 聚类命名落盘：cluster_names.json 存机械编号 T1..TK（auto_name=false 时即为最终标签）；
    # 命名提示(自动生成)另存 cluster_name_suggestions.json，用户目视解译后自行覆盖 cluster_names_manual。
    names_path = os.path.join(out_dir, "cluster_names.json")
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump(type_names, f, ensure_ascii=False, indent=2)
    print("    形态类型标签（机械编号，待你目视解译命名）：")
    for mt in sorted(type_names.keys()):
        print(f"      T{mt}")
    print(f"    ※ 自动命名提示见 cluster_name_suggestions.json；方向类特征只看数值看不出朝向，")
    print(f"      请结合 representative_buildings.csv 在 GIS 中查看实际形态 + cluster_profiles_radar.png 自行命名。")
    try:
        plot_naming_summary(suggestions, out_dir)
        print(f"    命名提示汇总表图已保存: {os.path.join(out_dir, 'cluster_naming_summary.png')}")
    except Exception as e:
        print(f"    [warn] 命名提示汇总表图生成失败: {e}")

    # 提取核心城市形态特征集（每类 top-N 合并去重），供下游「关键子图/解释」使用
    core_out = extract_core_features(df, feature_names, best_k, cfg, out_dir)

    print(f"\n=== 步骤⑤ 完成！核心形态子类 K={best_k}，产出: {out_dir} ===")
    print(f"    核心城市形态特征集 ({len(core_out['core_features'])} 个): {core_out['core_features']}")


if __name__ == "__main__":
    main()
