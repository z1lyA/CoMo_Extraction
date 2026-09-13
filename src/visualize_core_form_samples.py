#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_core_form_samples.py —— 每类选 N 个代表性建筑，平移到原点后叠加画在一张图，
便于肉眼比对核心形态类型（T1..TK）的形状差异。

数据来源：步骤⑤的 representative_buildings.csv（已按 dist_to_centroid 升序排序，
rank=1 即最接近簇心的「最典型」样本）。建筑真实轮廓来自 config.yaml 的 data.buildings_shp（按 building id 关联）。

为何平移到原点：建筑真实坐标分散在投影坐标系下（单位米），直接画会彼此远离、
挤在图边缘看不见；统一平移到各自质心原点后，形状大小/比例仍保留，可比性最强。

用法：
    python src/visualize_core_form_samples.py                 # 用最新一次 ⑤ 产物（在项目根执行）
    python src/visualize_core_form_samples.py --per-class 5  # 每类取 5 个（默认 5）
    python src/visualize_core_form_samples.py --clustering-dir <dir>  # 指定 ⑤ 目录
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
import glob
import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False


def find_latest_clustering(patterns, key_file):
    cands = []
    for pat in patterns:
        for d in glob.glob(pat):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, key_file)):
                cands.append(d)
    return max(cands, key=lambda d: os.path.getmtime(d)) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', nargs='?', default='config.yaml')
    ap.add_argument('--per-class', type=int, default=5, help='每类选取的代表建筑数（默认 5）')
    ap.add_argument('--clustering-dir', default=None, help='指定 ⑤ 输出目录（含 representative_buildings.csv）')
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"错误: 未找到配置文件 {args.config}")
        return
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    if not HAS_GEO:
        print("错误: 需要 geopandas，请在含 geopandas 的环境运行本脚本。")
        return

    RUN_DIR = os.environ.get('PIPELINE_RUN_DIR')
    clu_pat = [os.path.join(RUN_DIR, '04_Clustering', 'run_*')] if RUN_DIR else []
    clu_pat.append(os.path.join("results", "Clustering", "run_*_K*"))
    cdir = args.clustering_dir or find_latest_clustering(clu_pat, 'representative_buildings.csv')
    if cdir is None:
        print("错误: 未找到 ⑤ 的 representative_buildings.csv，请先运行步骤⑤。")
        return

    rep = pd.read_csv(os.path.join(cdir, 'representative_buildings.csv'))
    if 'building_ids_str' not in rep.columns or 'morph_type' not in rep.columns:
        print("错误: representative_buildings.csv 缺少 building_ids_str / morph_type 列。")
        return
    rep['bid'] = rep['building_ids_str'].astype(int)

    # 每类取 rank 最小的前 N 个（最典型）
    picked = {}
    for t in sorted(rep['morph_type'].unique()):
        sub = rep[rep['morph_type'] == t].sort_values('rank').head(args.per_class)
        picked[int(t)] = [int(b) for b in sub['bid'].tolist()]

    # 仅读取需要的建筑轮廓（where 过滤，避免加载全量 shp）
    form_shp = cfg['data']['buildings_shp']
    if not os.path.exists(form_shp):
        print(f"错误: 未找到建筑 shp: {form_shp}")
        return
    all_ids = sorted({b for v in picked.values() for b in v})
    id_field = None
    # 先探测 id 字段名：读一行看列
    probe = gpd.read_file(form_shp, rows=1)
    for c in probe.columns:
        if c.lower() in ('building_id', 'bid', 'id', 'buildingid'):
            id_field = c
            break
    if id_field is None:
        print("错误: 未在建筑 shp 找到 building id 字段，其列为:", list(probe.columns))
        return
    id_str = ",".join(str(i) for i in all_ids)
    try:
        gdf = gpd.read_file(form_shp, where=f"{id_field} IN ({id_str})")
    except Exception as e:
        print(f"    [warn] where 过滤失败({e})，回退读取全量 shp...")
        gdf = gpd.read_file(form_shp)
    gdf['bid'] = gdf[id_field].astype(int)
    geom_by_id = dict(zip(gdf['bid'], gdf.geometry))

    types = sorted(picked.keys())
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(10, 10))
    actual = {}
    for t in types:
        color = cmap((t - 1) % 10)
        cnt = 0
        for bid in picked[t]:
            geom = geom_by_id.get(bid)
            if geom is None:
                continue
            cx, cy = geom.centroid.x, geom.centroid.y
            g = geom.translate(-cx, -cy)
            if g.geom_type == 'Polygon':
                x, y = g.exterior.xy
                ax.plot(x, y, color=color, linewidth=1.0, alpha=0.7)
                cnt += 1
            elif g.geom_type == 'MultiPolygon':
                for part in g.geoms:
                    x, y = part.exterior.xy
                    ax.plot(x, y, color=color, linewidth=1.0, alpha=0.7)
                cnt += 1
        actual[t] = cnt

    ax.set_aspect('equal')
    ax.axhline(0, color='grey', lw=0.3)
    ax.axvline(0, color='grey', lw=0.3)
    handles = [Line2D([0], [0], color=cmap((t - 1) % 10), label=f"T{t} (n={actual[t]})") for t in types]
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.set_title(f"Core-form type shape gallery (top-{args.per_class} typical buildings per type, centered at origin)")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")

    out = os.path.join(cdir, f"core_form_samples_per_class{args.per_class}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[完成] 已保存: {out}")
    print("每类实际取到数 / 选取的 building_id:")
    for t in types:
        print(f"  T{t}: {actual[t]} / {picked[t]}")


if __name__ == "__main__":
    main()
