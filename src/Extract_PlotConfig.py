#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract_PlotConfig.py —— 步骤⑥：回贴全部建筑定地块构型（主导 / 多样）

对齐毕业论文 §3.2.2「地块级别的核心形态构型归纳」（论文 Step2）：
    利用 ⑤ 保存的 UMAP 降维器 + K-Means 模型，对「每个地块内的【所有建筑】」分类，
    统计各地块内各类核心形态建筑的构成比例，得到：
      - 主导型 (T_x-dom)：某类型占比 ≥ 阈值（论文统一 50%）且 建筑数 ≥ 下限
      - 多样型 (diverse) ：否则
    注：论文是对每个地块的全部建筑分类，而非仅 G_S 核心建筑；故本脚本读 examine.csv
        （步骤①导出的全量建筑特征）并用 ⑤ 模型套类型，覆盖全部 14919 个地块。

输入（自动发现最新产物）：
  - ⑤ results/Clustering/run_<ts>_K<k>/ 下：
        dimred_model.joblib   （UMAP/PCA 拟合模型）
        cluster_centroids.npy （K 个簇心，2D 嵌入空间）
        dimred_info.json      （feature_names / best_k / kind）
  - ① results/.../examine.csv（全部建筑：block_id, building_id, landuse_name + 23 维特征）

输出（results/PlotConfig/run_<ts>/）：
  - plot_config.csv            每地块: block_id, true_class_name, config_type, dominant_type, dominant_frac, num_buildings ...
  - plot_config_summary.json   统计与配置快照
  - plot_config.shp            （可选）按 block_id 关联 function.shp 的空间落图

参数（config.yaml 的 plot_config 段）：
  dominant_threshold_general: 0.50      一般地块主导阈值
  dominant_threshold_residential: 0.60  住宅地块主导阈值（更严）
  min_dominant_buildings: 2             主导型最少关键建筑数
  residential_class_names: ["居住"]     命中则套用更严阈值

输出（results/PlotConfig/run_<ts>/）：
  - plot_config.csv            每地块: block_id, true_class, config_type, dominant_type, dominant_frac, num_core_buildings ...
  - plot_config_summary.json   统计与配置快照
  - plot_config.shp            （可选）按 block_id 关联 function.shp 的空间落图
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
import json
import glob
import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

try:
    import joblib
except ImportError:
    joblib = None

try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# =============================================================================
# 工具
# =============================================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_latest(pattern, key_file=None):
    """返回含 key_file 且 mtime 最新者；pattern 可传单个 glob 串或串列表（多目录回退）。
    找不到返回 None。"""
    patterns = pattern if isinstance(pattern, (list, tuple)) else [pattern]
    cands = []
    for pat in patterns:
        for d in glob.glob(pat):
            if os.path.isdir(d):
                if key_file is None or os.path.exists(os.path.join(d, key_file)):
                    cands.append(d)
    if not cands:
        return None
    return max(cands, key=lambda d: os.path.getmtime(d))


def parse_primary_fid(token):
    """从 'blockid|buildingid' 取末段作为建筑主键（与 Cluster_CoreForms 一致）。"""
    s = str(token).strip()
    parts = [p.strip() for p in s.split('|') if str(p).strip() != '']
    if not parts:
        return None
    last = parts[-1]
    return int(last) if last.isdigit() else last


def classify_block(block, morph_lookup, general_thr, residential_thr, min_buildings, residential_names):
    """
    对单个地块回贴投票，返回构型记录 dict。
    block: ④ 的一个地块记录（含 G_S_building_ids / true_class_name）
    morph_lookup: {building_ids_str: morph_type}（来自 ⑤）
    """
    gs_ids = block.get('G_S_building_ids', [])
    morphs = [morph_lookup[bid] for bid in gs_ids if bid in morph_lookup]
    num_core = len(morphs)
    true_class = block.get('true_class', -1)
    true_name = block.get('true_class_name', str(true_class))
    is_res = true_name in residential_names
    thr = residential_thr if is_res else general_thr

    if num_core == 0:
        return {
            "graph_index": block.get("graph_index"),
            "block_id": block.get("block_id"),
            "true_class": true_class,
            "true_class_name": true_name,
            "is_residential": bool(is_res),
            "threshold_used": thr,
            "num_core_buildings": 0,
            "dominant_type": None,
            "dominant_count": 0,
            "dominant_frac": 0.0,
            "config_type": "no_core",
        }

    cnt = Counter(morphs)
    dom_type, dom_count = cnt.most_common(1)[0]
    dom_frac = dom_count / num_core
    if dom_count >= min_buildings and dom_frac >= thr:
        config_type = f"T{int(float(dom_type))}-dom"
    else:
        config_type = "diverse"
    return {
        "graph_index": block.get("graph_index"),
        "block_id": block.get("block_id"),
        "true_class": true_class,
        "true_class_name": true_name,
        "is_residential": bool(is_res),
        "threshold_used": thr,
        "num_core_buildings": num_core,
        "dominant_type": dom_type,
        "dominant_count": dom_count,
        "dominant_frac": round(dom_frac, 4),
        "config_type": config_type,
    }


# =============================================================================
# 全部建筑分类（论文 Step2：用已保存 UMAP/KMeans 模型套全量建筑）
# =============================================================================
def classify_all_buildings(examine_csv, dimred_model, centroids, feature_names, chunk_size=50000):
    """论文 Step2：用 ⑤ 保存的降维器+KMeans 簇心，对 examine.csv 中的【全部建筑】分类，
    再按 block_id 聚合每地块的类型构成。返回 (block_counts, block_landuse)。
    block_counts: {block_id: Counter({T: 数量})}；block_landuse: {block_id: 用地名}
    """
    block_counts = defaultdict(Counter)
    block_landuse = {}
    cols = ['block_id', 'building_id', 'landuse_name'] + list(feature_names)
    cent = np.array(centroids, dtype=float)  # (K, 2)
    n_total = 0
    for chunk in pd.read_csv(examine_csv, usecols=cols, chunksize=chunk_size):
        X = chunk[feature_names].values.astype(float)
        emb = dimred_model.transform(X)                          # (n,2)
        d = ((emb[:, None, :] - cent[None, :, :]) ** 2).sum(2)   # (n,K) 欧氏距离平方
        labels = d.argmin(1) + 1                                # 1..K
        bids = chunk['block_id'].values
        lids = chunk['landuse_name'].values
        for bid, lid, lab in zip(bids, lids, labels):
            block_counts[bid][int(lab)] += 1
            if bid not in block_landuse:
                block_landuse[bid] = lid
        n_total += len(chunk)
    print(f"    已对 {n_total} 栋建筑完成类型归属（覆盖 {len(block_counts)} 个地块）")
    return block_counts, block_landuse


# =============================================================================
# 住宅地块判定：兼容标签的两种表示（数值 Level1 或中文功能名）
# =============================================================================
CLASS_ID_TO_CN = {
    1.0: "居住", 2.0: "商业", 3.0: "工业", 4.0: "交通", 5.0: "公共管理与服务",
}


def _is_residential(landuse_val, residential_names):
    """判断某地块是否住宅用地：既支持 config 里写中文名（如「居住」），
    也支持数据标签为数值（Level1=1.0）的写法，两种表示都能正确命中。"""
    v = str(landuse_val)
    if v in residential_names:
        return True
    try:
        cn = CLASS_ID_TO_CN.get(float(landuse_val))
    except (TypeError, ValueError):
        cn = None
    return bool(cn is not None and cn in residential_names)


# =============================================================================
# 主流程
# =============================================================================
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    if not os.path.exists(config_path):
        print(f"错误: 未找到配置文件 {config_path}")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    pc = cfg.get('plot_config', {})
    general_thr = float(pc.get('dominant_threshold_general', 0.50))
    residential_thr = float(pc.get('dominant_threshold_residential', 0.60))
    min_buildings = int(pc.get('min_dominant_buildings', 2))
    residential_names = set(pc.get('residential_class_names', ["居住"]))

    # ---- 自动发现最新 ⑤ 产物（优先当前运行目录，回退历史 results/）----
    RUN_DIR = os.environ.get('PIPELINE_RUN_DIR')
    clu_pat = [os.path.join(RUN_DIR, '04_Clustering', 'run_*')] if RUN_DIR else []
    clu_pat.append(os.path.join("results", "Clustering", "run_*_K*"))
    clustering_dir = pc.get('clustering_dir') or find_latest(clu_pat, key_file="dimred_model.joblib")
    if clustering_dir is None:
        print("错误: 未找到 ⑤ 的 dimred_model.joblib，请先运行步骤⑤。")
        return
    dimred_path = os.path.join(clustering_dir, "dimred_model.joblib")
    cent_path = os.path.join(clustering_dir, "cluster_centroids.npy")
    info_path = os.path.join(clustering_dir, "dimred_info.json")
    if not (os.path.exists(dimred_path) and os.path.exists(cent_path) and os.path.exists(info_path)):
        print("错误: ⑤ 未完整保存降维模型/簇心/元信息，无法对全量建筑分类。请重跑步骤⑤。")
        return
    if joblib is None:
        print("错误: joblib 不可用，无法加载 dimred_model.joblib（请 pip install joblib）。")
        return
    dimred_model = joblib.load(dimred_path)
    centroids = np.load(cent_path).tolist()
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    feature_names = info['feature_names']
    best_k = int(info['best_k'])
    print(f"=== 步骤⑥ 回贴（论文Step2：全部建筑分类→地块构型）===")
    print(f"  ⑤ 来源: {clustering_dir}  (降维={info['kind']}, K={best_k})")
    print(f"  阈值: 一般={general_thr} 住宅={residential_thr} 主导最少建筑={min_buildings}")

    # ---- 发现 examine.csv（全部建筑特征，来自步骤① Build_Graphs）----
    exm_pat = [os.path.join(RUN_DIR, '01_Graphs', '*', 'examine.csv')] if RUN_DIR else []
    exm_pat.append(os.path.join("results", "Graphs", "*", "examine.csv"))
    examine_csv = pc.get('examine_csv')
    if examine_csv is None:
        _cands = []
        for _pat in exm_pat:
            for _fp in glob.glob(_pat):
                if os.path.isfile(_fp):
                    _cands.append(_fp)
        examine_csv = max(_cands, key=lambda p: os.path.getmtime(p)) if _cands else None
    if examine_csv is None or not os.path.exists(examine_csv):
        print("错误: 未找到 examine.csv（步骤① Build_Graphs 应导出全部建筑特征）。请先运行步骤①。")
        return
    print(f"  全部建筑来源: {examine_csv}")

    # ---- 对全部建筑分类并聚合到地块（论文 Step2：用已保存 UMAP/KMeans 套全量建筑）----
    block_counts, block_landuse = classify_all_buildings(examine_csv, dimred_model, centroids, feature_names)

    # ---- 逐地块判定主导 / 多样 ----
    rows = []
    type_counter = Counter()
    for bid, cnt in block_counts.items():
        n = sum(cnt.values())
        if n == 0:
            continue
        dom_type, dom_count = cnt.most_common(1)[0]
        dom_frac = dom_count / n
        is_res = _is_residential(block_landuse.get(bid), residential_names)
        thr = residential_thr if is_res else general_thr
        if dom_count >= min_buildings and dom_frac > thr:
            config_type = f"T{int(dom_type)}-dom"
        else:
            config_type = "diverse"
        rows.append({
            "block_id": bid,
            "true_class_name": block_landuse.get(bid),
            "is_residential": bool(is_res),
            "threshold_used": thr,
            "num_buildings": int(n),
            "dominant_type": int(dom_type),
            "dominant_count": int(dom_count),
            "dominant_frac": round(dom_frac, 4),
            "config_type": config_type,
        })
        type_counter[config_type] += 1

    out_df = pd.DataFrame(rows)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(RUN_DIR, '05_PlotConfig', f"run_{ts}") if RUN_DIR else os.path.join("results", "PlotConfig", f"run_{ts}")
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, "plot_config.csv")
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "timestamp": ts,
        "clustering_source": clustering_dir,
        "examine_source": examine_csv,
        "thresholds": {
            "general": general_thr,
            "residential": residential_thr,
            "min_dominant_buildings": min_buildings,
        },
        "residential_class_names": list(residential_names),
        "num_blocks": int(len(out_df)),
        "config_type_counts": dict(type_counter),
        "dominant_counts": {k: int(v) for k, v in type_counter.items() if k.endswith("-dom")},
        "diverse_count": int(type_counter.get("diverse", 0)),
        "no_core_count": int(type_counter.get("no_core", 0)),
    }
    summary_path = os.path.join(out_dir, "plot_config_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n地块构型统计（共 {len(out_df)} 个地块）:")
    for k, v in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"  {k:14s}: {v}")
    print(f"[完成] 已保存: {csv_path}")
    print(f"[完成] 已保存: {summary_path}")

    # 构型分布图（英文标签，无乱码风险）
    if HAS_PLOT:
        try:
            counts = dict(type_counter)
            labels = sorted(counts.keys(), key=lambda k: -counts[k])
            vals = [counts[k] for k in labels]
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#378ADD" if k == "diverse" else "#639922" for k in labels]
            ax.bar(range(len(labels)), vals, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel("plot count")
            ax.set_title("Plot configuration type distribution")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.5, str(v), ha='center', fontsize=8)
            fig.tight_layout()
            dist_p = os.path.join(out_dir, "plot_config_distribution.png")
            fig.savefig(dist_p, dpi=300); plt.close(fig)
            print(f"[完成] 构型分布图已保存: {dist_p}")
        except Exception as e:
            print(f"    [warn] 构型分布图生成失败: {e}")

    # ---- 可选：空间落图（按 block_id 关联 function.shp）----
    if HAS_GEO and pc.get('extract_shp', False):
        landuse_shp = cfg['data'].get('landuse_shp')
        if landuse_shp and os.path.exists(landuse_shp):
            try:
                lugdf = gpd.read_file(landuse_shp)
                bid_col = cfg['data'].get('landuse_block_id_field', 'block_id')
                if bid_col in lugdf.columns:
                    merged = lugdf.merge(out_df, left_on=bid_col, right_on='block_id', how='inner')
                    if not merged.empty:
                        out_shp = os.path.join(out_dir, "plot_config.shp")
                        merged.to_file(out_shp)
                        print(f"[完成] 地块构型空间图已保存: {out_shp} ({len(merged)} 个地块)")
                    else:
                        print("    [warn] block_id 关联 0 行，跳过 shp 导出。")
                else:
                    print(f"    [warn] function.shp 无 {bid_col} 字段，跳过 shp 导出。")
            except Exception as e:
                print(f"    [warn] shp 导出失败: {e}")
        else:
            print("    [跳过] 未配置/未找到 landuse_shp，跳过 shp 导出。")
    else:
        print("    [跳过] 未启用或 geopandas 不可用，跳过 shp 导出。")

    print(f"\n=== 步骤⑥ 完成！产出: {out_dir} ===")


if __name__ == "__main__":
    main()
