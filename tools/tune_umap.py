#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UMAP 超参自动扫描 (tune_umap)

遍历 (n_neighbors, min_dist) 候选组合，每组跑真 UMAP + K 自动搜索（min_k..max_k），
用 silhouette 系数量化「点分得多开」，输出：
  - umap_tune_results.csv   对比表（n_neighbors, min_dist, best_K, best_silhouette）
  - umap_tune_grid.png      网格散点对比图（每格按最优 K 着色 + 标注 silhouette）
  - 终端打印推荐组合（不自动写回 config，仅给建议值）

独立脚本，复用现有 run 的 ④⑤ 产物（graph_dataset.pt + gnnexplainer_core.json + fs_global.json），
无需重跑训练/解释。把"肉眼调 UMAP 让图分开"变成"量化选优 + 可视化对比"，可复现、发布时别人能看懂。

用法：
  python tools/tune_umap.py [config.yaml]      # 真实运行（吃现有 run 产物，在项目根执行）
  python tools/tune_umap.py --smoke            # 合成数据 smoke test（非真实，仅验证链路）
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
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)                              # 让 src 作为包被加载

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.Cluster_CoreForms import (
    load_gs_building_ids,
    find_latest_gnn_explainer_core,
    find_latest_fs_global,
    kmeans_numpy,
    silhouette_numpy,
    zscore_matrix,
    largest_cluster_frac,
    run_umap_safe,
    ensure_dir,
)


def collect_gs_features(cfg, run_dir):
    """复用步骤⑤ main 的 G_S 核心建筑特征收集逻辑。返回 (feat_mat, feature_names, df)。"""
    import torch
    import pickle
    cl = cfg['clustering']
    graph_dir = cfg['training_paths']['graph_data_dir']
    gd_path = os.path.join(graph_dir, "graph_dataset.pt")
    meta_path = os.path.join(graph_dir, "metadata.pkl")
    if not (os.path.exists(gd_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"图数据不存在: {gd_path} / {meta_path}（请先跑步骤①② 或指定正确的 graph_data_dir）")
    graphs = torch.load(gd_path, map_location='cpu', weights_only=False)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    all_feature_names = meta.get('feature_names', [])
    class_names = meta.get('class_names', [])

    feature_source = str(cl.get('feature_source', 'fs_from_explainer')).lower()
    if feature_source == 'fs_from_explainer':
        fs_global = find_latest_fs_global(run_dir)
        sel = fs_global.get('selected_features', []) if fs_global else []
        feature_names = [f for f in sel if f in all_feature_names]
        if not feature_names:
            raise ValueError("fs_global.json 的 selected_features 为空或与图特征不匹配，无法聚类。")
    elif feature_source == 'all':
        feature_names = list(all_feature_names)
    else:
        feature_names = [f for f in cl.get('custom_features', []) if f in all_feature_names] or list(all_feature_names)

    use_gs = bool(cl.get('use_gs_buildings', True))
    gs_set = None
    if use_gs:
        core_path = find_latest_gnn_explainer_core(run_dir)
        if core_path:
            gs_set = load_gs_building_ids(core_path)

    rows = []
    for gi, g in enumerate(graphs):
        x = g.x.cpu().numpy()
        bstrs = getattr(g, 'building_ids_str', [''] * x.shape[0])
        yval = int(g.y.item()) if hasattr(g.y, 'item') else int(g.y)
        bid = getattr(g, 'block_id', -1)
        for ni in range(x.shape[0]):
            bid_str = bstrs[ni] if ni < len(bstrs) else ''
            if gs_set is not None and not any(p in gs_set for p in str(bid_str).split('|')):
                continue
            row = {'graph_index': gi, 'node_index': ni, 'block_id': bid,
                   'class_id': yval, 'building_ids_str': bid_str}
            for fname in feature_names:
                row[fname] = float(x[ni, all_feature_names.index(fname)])
            rows.append(row)

    df = pd.DataFrame(rows)
    feat_mat = df[feature_names].values.astype(float)
    if not np.isfinite(feat_mat).all():
        col_med = np.nanmedian(feat_mat, axis=0)
        feat_mat = np.where(np.isfinite(feat_mat), feat_mat, col_med)
    return feat_mat, feature_names, df


def run_scan(feat_mat, feat_eval, nn_grid, md_grid, min_k, max_k, rs):
    """对每 (nn, md) 跑 UMAP + K 搜索，返回 results 列表（含 emb 缓存供绘图复用）。

    【评估空间解耦】聚类在 UMAP 嵌入空间做，但选优用的轮廓系数一律在
    feat_eval（原始 F_S 标准化特征空间）上算。否则 min_dist 越小 → 嵌入空间簇越紧
    → 分数越高，等于「用被超参掰弯的尺子量身高」，扫描必然退回极小 min_dist。
    另同时输出嵌入空间分数（仅出图观感参考）与最大簇占比（识别退化解）。
    """
    results = []
    for nn in nn_grid:
        for md in md_grid:
            emb, err = run_umap_safe(feat_mat, nn, md, rs, timeout=900)
            if emb is None:
                print(f"    [跳过] n_neighbors={nn}, min_dist={md} UMAP 失败: {err}")
                results.append({'n_neighbors': nn, 'min_dist': md, 'best_K': -1,
                                'silhouette_raw': float('nan'), 'silhouette_embedding': float('nan'),
                                'largest_cluster_frac': float('nan'), 'emb': None})
                continue
            best_k, best_sc, best_emb_sc, best_frac = min_k, -1.0, float('nan'), float('nan')
            for k in range(min_k, max_k + 1):
                labels = kmeans_numpy(emb, k, random_state=rs, n_init=10)
                sc = silhouette_numpy(feat_eval, labels, random_state=rs)   # 主判据
                if sc > best_sc:
                    best_sc, best_k = sc, k
                    best_emb_sc = silhouette_numpy(emb, labels, random_state=rs)
                    best_frac = largest_cluster_frac(labels)
            results.append({'n_neighbors': nn, 'min_dist': md, 'best_K': best_k,
                            'silhouette_raw': best_sc, 'silhouette_embedding': best_emb_sc,
                            'largest_cluster_frac': best_frac, 'emb': emb})
            flag = "  [警告]退化(单簇占比过高)" if best_frac >= 0.90 else ""
            print(f"    n_neighbors={nn:>2}, min_dist={md:<7} -> best_K={best_k:>2}  "
                  f"silhouette(原始空间)={best_sc:.4f}  [嵌入空间={best_emb_sc:.4f} 仅参考]  "
                  f"最大簇占比={best_frac:.1%}{flag}")
    return results


def plot_grid(results, nn_grid, md_grid, out_path):
    nrows, ncols = len(nn_grid), len(md_grid)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    for i, nn in enumerate(nn_grid):
        for j, md in enumerate(md_grid):
            ax = axes[i][j]
            r = next((r for r in results if r['n_neighbors'] == nn and r['min_dist'] == md), None)
            if r and r.get('emb') is not None and r['best_K'] > 0:
                emb = r['emb']
                labels = kmeans_numpy(emb, r['best_K'], random_state=42, n_init=10)
                ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=3, cmap='tab10', alpha=0.6)
                ax.set_title(f"nn={nn}, md={md}\nK={r['best_K']}, sil(raw)={r['silhouette_raw']:.3f}")
            else:
                ax.set_title(f"nn={nn}, md={md}\n(UMAP 失败)")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = sys.argv[1:]
    smoke = '--smoke' in args
    config_path = next((a for a in args if a != '--smoke' and a.endswith('.yaml')), 'config.yaml')
    config_path = os.path.join(_ROOT, config_path) if not os.path.isabs(config_path) else config_path

    if smoke:
        print("=== UMAP 超参扫描 SMOKE TEST（合成数据，非真实）===")
        rng = np.random.default_rng(42)
        feat_mat = rng.standard_normal((500, 10)).astype(float)
        feat_mat[:, 0] *= 50.0
        feat_mat[:, 1] = rng.uniform(0, 360, 500)
        nn_grid, md_grid = [15, 30], [0.05, 0.25]
        min_k, max_k, rs = 3, 12, 42
    else:
        import yaml
        if not os.path.exists(config_path):
            print(f"错误: 未找到配置文件 {config_path}")
            return
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        cl = cfg['clustering']
        run_dir = os.environ.get('PIPELINE_RUN_DIR')
        tune_cfg = cl.get('tune', {}) or {}
        nn_grid = tune_cfg.get('n_neighbors_grid', [15, 30])
        md_grid = tune_cfg.get('min_dist_grid', [0.05, 0.25])
        min_k = int(cl['min_k'])
        max_k = int(cl['max_k'])
        rs = int(cl['random_state'])
        print("=== UMAP 超参自动扫描 (tune_umap) ===")
        feat_mat, feature_names, df = collect_gs_features(cfg, run_dir)
        print(f"    核心建筑特征矩阵: {feat_mat.shape} (建筑数 × {len(feature_names)} 特征)")
        print(f"    扫描网格: n_neighbors={nn_grid} × min_dist={md_grid} = {len(nn_grid) * len(md_grid)} 组")
        print(f"    K 搜索范围: {min_k}..{max_k}  |  选优判据: 原始 F_S 标准化空间轮廓系数")

    feat_eval = zscore_matrix(feat_mat)   # 所有超参共用的固定标尺
    print(f"    开始扫描 {len(nn_grid) * len(md_grid)} 组 UMAP 超参...")
    results = run_scan(feat_mat, feat_eval, nn_grid, md_grid, min_k, max_k, rs)

    cols = ('n_neighbors', 'min_dist', 'best_K', 'silhouette_raw',
            'silhouette_embedding', 'largest_cluster_frac')
    res_df = pd.DataFrame([{k: r[k] for k in cols} for r in results])
    res_df = res_df.sort_values('silhouette_raw', ascending=False)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if smoke:
        out_dir = os.path.join(_ROOT, "results", f"tune_umap_smoke_{ts}")
    else:
        run_dir = os.environ.get('PIPELINE_RUN_DIR')
        out_dir = (os.path.join(run_dir, '04_Clustering', f"tune_umap_{ts}")
                   if run_dir else os.path.join(_ROOT, "results", f"tune_umap_{ts}"))
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "umap_tune_results.csv")
    res_df.to_csv(csv_path, index=False)
    grid_path = os.path.join(out_dir, "umap_tune_grid.png")
    plot_grid(results, nn_grid, md_grid, grid_path)

    print("\n=== 扫描结果汇总（按原始空间 silhouette 降序）===")
    print(res_df.to_string(index=False))

    # 优先在非退化解里挑；全退化时再放开
    ok = res_df[(res_df['largest_cluster_frac'] < 0.90) & res_df['silhouette_raw'].notna()]
    pool = ok if not ok.empty else res_df[res_df['silhouette_raw'].notna()]
    if not pool.empty:
        best = pool.iloc[0]
        if ok.empty:
            print("\n[警告] 所有组合都出现「单簇占比≥90%」的退化解，下方推荐仅供参考，建议扩大网格或检查特征。")
        print(f"\n[完成] 推荐: umap_n_neighbors={int(best['n_neighbors'])}, umap_min_dist={best['min_dist']}")
        print(f"   原始空间 silhouette={best['silhouette_raw']:.4f} | best_K={int(best['best_K'])} | "
              f"最大簇占比={best['largest_cluster_frac']:.1%} | 嵌入空间={best['silhouette_embedding']:.4f}(仅参考)")
        print(f"   产物目录: {out_dir}")
        if not smoke:
            print("   （未自动写回 config；请手动修改 config.yaml 的 clustering.umap_n_neighbors / "
                  "umap_min_dist 后重跑步骤⑤⑥）")
    else:
        print("\n[警告] 所有组合 UMAP 均失败，请检查环境（numba/umap）或调小数据量。")


if __name__ == "__main__":
    main()
