"""
Build_Graphs.py
城市形态图构建器（优化实验版 / opt）—— 22 维特征 + 距离边权 + building_ids + 自动检查导出

相对 2build_graphs22.py 的改动：
  [P2] distance_threshold 由配置统一控制，默认 200m（依据 Liu & Song 2025 Nanjing Walled City 论文）。
  [P1] 把每个地块的建筑质心（centroids）一并存入 PyG Data，供训练阶段按真实地理距离实时计算边权，
       无需在构图时预存边权（边权在 GNN 内逐层随 SAGPooling 重算，更准确）。
  [P0] 构図完成后自动把输出目录写回 config.yaml（统一配置文件，修复旧版接链 bug）。
  [修复] docstring 维度 23 → 22。

其余特征计算（22 维形态指标）、质心合并、Voronoi 面积、检查导出等逻辑与旧版一致。
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
import yaml
import json
import math
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data
from scipy.spatial import Delaunay, KDTree, Voronoi
import shapely as _shapely
from shapely.ops import unary_union
from shapely.geometry import Polygon, Point
import warnings
import networkx as nx
import pickle
import datetime

warnings.filterwarnings('ignore')


# ============================================================
#   辅助函数
# ============================================================

def _normalize_angle_pi(angle):
    while angle < 0:
        angle += np.pi
    while angle >= np.pi:
        angle -= np.pi
    return float(angle)


def _get_polygon_coords(geometry):
    if geometry.geom_type == 'Polygon':
        return np.array(geometry.exterior.coords)[:-1]
    elif geometry.geom_type == 'MultiPolygon':
        largest = max(geometry.geoms, key=lambda g: g.area)
        return np.array(largest.exterior.coords)[:-1]
    return np.array([]).reshape(0, 2)


def _get_convex_hull_coords(geometry):
    ch = geometry.convex_hull
    return _get_polygon_coords(ch), ch


def _compute_longest_chord(ch_coords):
    n = len(ch_coords)
    if n < 2:
        return 0.0, 0.0, (None, None)
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(ch_coords))
    max_i, max_j = np.unravel_index(np.argmax(dists), dists.shape)
    max_dist = dists[max_i, max_j]
    vec = ch_coords[max_j] - ch_coords[max_i]
    angle = _normalize_angle_pi(np.arctan2(vec[1], vec[0]))
    return float(max_dist), float(angle), (int(max_i), int(max_j))


def _compute_minimum_bounding_rectangle(geometry):
    try:
        mbr = geometry.minimum_rotated_rectangle
        coords = np.array(mbr.exterior.coords)[:-1]
        if len(coords) < 4:
            return 1.0, 1.0, 0.0
        edge1 = np.linalg.norm(coords[1] - coords[0])
        edge2 = np.linalg.norm(coords[2] - coords[1])
        if edge1 >= edge2:
            l_sbr, w_sbr = edge1, edge2
            vec = coords[1] - coords[0]
        else:
            l_sbr, w_sbr = edge2, edge1
            vec = coords[2] - coords[1]
        angle = _normalize_angle_pi(np.arctan2(vec[1], vec[0]))
        return float(l_sbr), float(w_sbr), float(angle)
    except Exception:
        return 1.0, 1.0, 0.0


def _compute_bisector_orientation(ch_coords, chord_indices, lco_angle):
    """
    依据最长弦两端点构造三角形，并取夹角平分线方向的近似实现。
    若无法稳定计算，则退化为最长弦垂线方向。
    """
    if len(ch_coords) < 3 or chord_indices[0] is None or chord_indices[1] is None:
        return _normalize_angle_pi(lco_angle + np.pi / 2)

    i, j = chord_indices
    p1 = ch_coords[i]
    p2 = ch_coords[j]

    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)
    if line_norm == 0:
        return _normalize_angle_pi(lco_angle + np.pi / 2)

    best_k = None
    best_dist = -1.0
    for k in range(len(ch_coords)):
        if k == i or k == j:
            continue
        pk = ch_coords[k]
        # [修复] 显式计算 2D 向量叉积（z 分量 = a_x*b_y - a_y*b_x），
        #        兼容 numpy 2.x（旧版 np.cross 对 2D 向量会隐式按 3D 处理，新版直接报错）。
        cross_2d = line_vec[0] * (pk[1] - p1[1]) - line_vec[1] * (pk[0] - p1[0])
        dist = abs(cross_2d) / line_norm
        if dist > best_dist:
            best_dist = dist
            best_k = k

    if best_k is None:
        return _normalize_angle_pi(lco_angle + np.pi / 2)

    apex = ch_coords[best_k]
    v1 = p1 - apex
    v2 = p2 - apex
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return _normalize_angle_pi(lco_angle + np.pi / 2)

    bisector = v1 / n1 + v2 / n2
    if np.linalg.norm(bisector) == 0:
        return _normalize_angle_pi(lco_angle + np.pi / 2)

    return _normalize_angle_pi(np.arctan2(bisector[1], bisector[0]))


def _compute_weighted_orientation(coords):
    """严格按论文表述实现：Σ(edge_i × Ori_i) / Σ(edge_i)"""
    n = len(coords)
    if n < 2:
        return 0.0
    p1 = coords
    p2 = np.roll(coords, -1, axis=0)
    vecs = p2 - p1
    lengths = np.linalg.norm(vecs, axis=1)
    valid = lengths > 0
    if not np.any(valid):
        return 0.0
    vecs = vecs[valid]
    lengths = lengths[valid]
    angles = np.array([_normalize_angle_pi(np.arctan2(v[1], v[0])) for v in vecs])
    total_len = np.sum(lengths)
    if total_len == 0:
        return 0.0
    return float(np.sum(lengths * angles) / total_len)


def _compute_max_circularity(area, distances):
    r_max = np.max(distances) if len(distances) > 0 else 1.0
    if r_max == 0:
        return 1.0
    return float(np.sqrt(area / np.pi) / r_max)


def _compute_bci(distances):
    """按用户确认出处中的原式实现，不再除以 n。"""
    n = len(distances)
    if n == 0:
        return 0.0
    total = np.sum(distances)
    if total <= 0:
        return 0.0
    percentages = (distances / total) * 100.0
    expected = 100.0 / n
    bci = np.sum(np.abs(percentages - expected))
    return float(bci)


def _minimum_enclosing_circle(points):
    """
    最小外接圆（用于 DCM 的 ASCC）。

    使用 shapely 的 GEOS 实现（minimum_bounding_circle），期望 O(n)。
    相比原版朴素 O(V^4) 枚举点对/三点+包含检测，对 SinoBF 这种顶点多的大数据
    可快几个数量级（11.5h -> 预计 2-3h）。

    进一步先取凸包：多边形 MEC == 顶点集 MEC == 凸包顶点 MEC，
    顶点数从几百压到十几，再喂给 GEOS，进一步提速。
    """
    from shapely.geometry import MultiPoint
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.array([0.0, 0.0]), 0.0
    if len(pts) == 1:
        return pts[0], 0.0

    hull = MultiPoint(pts).convex_hull
    if hull.is_empty:
        centroid = np.mean(pts, axis=0)
        return centroid, float(np.max(np.linalg.norm(pts - centroid, axis=1)))

    circle = _shapely.minimum_bounding_circle(hull)  # GEOS C 实现，期望 O(n)
    center = np.array([circle.centroid.x, circle.centroid.y], dtype=float)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
    return center, radius


def _compute_dcm(coords, area):
    center, radius = _minimum_enclosing_circle(coords)
    if radius <= 0:
        return 1.0
    a_scc = np.pi * (radius ** 2)
    if a_scc <= 0:
        return 1.0
    return float(area / a_scc)


def _compute_ellipticity(coords, lco_angle, l_max):
    if l_max <= 0 or len(coords) < 2:
        return 1.0
    perp_angle = lco_angle + np.pi / 2
    unit = np.array([np.cos(perp_angle), np.sin(perp_angle)])
    projections = coords @ unit
    lwidth = float(np.max(projections) - np.min(projections))
    if l_max <= 0:
        return 1.0
    return float(lwidth / l_max)


def _compute_exchange_index(area, geometry):
    """按定义真实计算建筑与等面积圆的交面积。"""
    if area <= 0 or geometry.is_empty:
        return 0.0
    radius = math.sqrt(area / math.pi)
    center = geometry.centroid
    eac = Point(center.x, center.y).buffer(radius, resolution=64)
    a_eac = eac.area
    if a_eac <= 0:
        return 0.0
    inter_area = geometry.intersection(eac).area
    return float(1.0 - inter_area / a_eac)


def _compute_cross_moment(coords, cx, cy):
    x_c = coords[:, 0] - cx
    y_c = coords[:, 1] - cy
    return float(np.sum(x_c * y_c))


def _compute_variance_spreading(coords, cx, cy):
    """Variance (spreading)：论文 Table 1 shape 指标。
    顶点相对质心在 x、y 方向的方差之和 = [Σ(x_i-cx)² + Σ(y_i-cy)²] / n，
    描述建筑轮廓顶点相对质心的铺展程度（Jähne, 2013）。
    """
    if len(coords) == 0:
        return 0.0
    x_c = coords[:, 0] - cx
    y_c = coords[:, 1] - cy
    n = len(coords)
    return float((np.sum(x_c ** 2) + np.sum(y_c ** 2)) / n)


def _compute_voronoi_areas(all_centroids_xy, block_polygon):
    n = len(all_centroids_xy)
    if n == 0:
        return np.array([], dtype=float)
    if block_polygon is None or block_polygon.is_empty:
        return np.ones(n, dtype=float)
    if n < 4:
        return np.full(n, block_polygon.area / max(n, 1), dtype=float)
    try:
        bounds = block_polygon.bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.5
        mirror_points = np.array([
            [bounds[0] - margin, bounds[1] - margin],
            [bounds[2] + margin, bounds[1] - margin],
            [bounds[0] - margin, bounds[3] + margin],
            [bounds[2] + margin, bounds[3] + margin],
        ])
        all_pts = np.vstack([all_centroids_xy, mirror_points])
        vor = Voronoi(all_pts)
        areas = np.full(n, block_polygon.area / n, dtype=float)
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if (not region) or (-1 in region):
                continue
            poly = Polygon(vor.vertices[region])
            clipped = poly.intersection(block_polygon)
            if not clipped.is_empty and clipped.area > 0:
                areas[i] = float(clipped.area)
        return areas
    except Exception:
        return np.full(n, block_polygon.area / max(n, 1), dtype=float)


def export_graph_examination(pyg_dataset, out_dir, feature_names, label_encoder):
    all_features = []
    graph_level_stats = []

    for i, graph in enumerate(tqdm(pyg_dataset, desc='导出检查结果',
                                    disable=not sys.stdout.isatty())):
        block_id = getattr(graph, 'block_id', i)
        y = getattr(graph, 'y', -1)
        if isinstance(y, torch.Tensor):
            y = y.view(-1)[0].item() if y.numel() > 0 else -1
        landuse_name = f'类别_{y}'
        if label_encoder is not None:
            try:
                landuse_name = label_encoder.inverse_transform([y])[0]
            except Exception:
                pass

        x = graph.x.detach().cpu().numpy()
        building_ids_str = getattr(graph, 'building_ids_str', None)
        if building_ids_str is None:
            building_ids_str = [''] * x.shape[0]

        graph_level_stats.append({
            'graph_id': i,
            'block_id': block_id,
            'num_nodes': int(x.shape[0]),
            'landuse_label': int(y),
            'landuse_name': str(landuse_name),
        })

        for node_idx in range(x.shape[0]):
            row = {
                'graph_id': i,
                'block_id': block_id,
                'node_idx': node_idx,
                'building_id': building_ids_str[node_idx] if node_idx < len(building_ids_str) else '',
                'landuse_label': int(y),
                'landuse_name': str(landuse_name),
            }
            for feat_idx in range(x.shape[1]):
                row[feature_names[feat_idx]] = float(x[node_idx, feat_idx])
            all_features.append(row)

    df = pd.DataFrame(all_features)
    basic_cols = ['graph_id', 'block_id', 'node_idx', 'building_id', 'landuse_label', 'landuse_name']
    feature_cols = [c for c in df.columns if c not in basic_cols]
    df = df[basic_cols + feature_cols]
    df.to_csv(os.path.join(out_dir, 'examine.csv'), index=False, encoding='utf-8-sig')

    summary_rows = []
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors='coerce')
        summary_rows.append({
            'feature_name': col,
            'count': int(s.notna().sum()),
            'mean': float(s.mean()),
            'std': float(s.std(ddof=0)),
            'min': float(s.min()),
            'max': float(s.max()),
            'n_unique': int(s.nunique(dropna=True)),
            'is_constant_like': bool((s.nunique(dropna=True) <= 1) or (float(s.std(ddof=0)) == 0.0)),
        })
    pd.DataFrame(summary_rows).sort_values(['is_constant_like', 'n_unique', 'std'], ascending=[False, True, True]).to_csv(
        os.path.join(out_dir, 'feature_summary.csv'), index=False, encoding='utf-8-sig'
    )

    pd.DataFrame(graph_level_stats).to_csv(os.path.join(out_dir, 'graph_summary.csv'), index=False, encoding='utf-8-sig')

    run_info = {
        'output_dir': out_dir,
        'graph_count': int(len(pyg_dataset)),
        'node_record_count': int(len(df)),
        'feature_count': int(len(feature_cols)),
        'feature_names': feature_cols,
    }
    with open(os.path.join(out_dir, 'examine_run_info.json'), 'w', encoding='utf-8') as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)


# ============================================================
#   数据清洗模块（移植自 temp/2026-07-27/1build_clean2.py v2.0）
#   作用：把"生建筑形态"(raw_buildings_shp) 清洗为可供构图的数据。
#   步骤：[0·可选]合并重叠/接触建筑 → [1]几何有效性+类型 → [2]面积阈值 → [3]伸长度(线状)过滤
#   所有参数由 config.yaml 的 cleaning 段控制（应对不同数据源/出版方差异）。
# ============================================================

def calculate_elongation(geom):
    """计算多边形最小外接矩形的长宽比（伸长度）。"""
    try:
        rect = geom.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        if len(coords) < 4:
            return 1.0
        edge1 = math.sqrt((coords[0][0] - coords[1][0]) ** 2 + (coords[0][1] - coords[1][1]) ** 2)
        edge2 = math.sqrt((coords[1][0] - coords[2][0]) ** 2 + (coords[1][1] - coords[2][1]) ** 2)
        if edge1 == 0 or edge2 == 0:
            return float('inf')
        return max(edge1, edge2) / min(edge1, edge2)
    except Exception:
        return 1.0


def merge_overlapping_buildings(gdf, fid_column, height_column='Height',
                                form_id_column='form_id', buffer_distance=0.1):
    """合并空间上重叠或接触的建筑；Height 取 max，form_id 取 first。"""
    print(f"\n[清洗步骤 0] 合并重叠/接触建筑...")
    original_count = len(gdf)
    if original_count == 0:
        return gdf
    if height_column not in gdf.columns:
        gdf[height_column] = 0
    if form_id_column not in gdf.columns:
        gdf[form_id_column] = gdf[fid_column]

    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf_buffered = gdf.copy()
    gdf_buffered['geometry'] = gdf.geometry.buffer(buffer_distance, cap_style=3, join_style=3)

    sjoined = gpd.sjoin(gdf_buffered.set_index(fid_column), gdf.set_index(fid_column),
                        how='left', predicate='intersects', lsuffix='l', rsuffix='r')
    G = nx.Graph()
    G.add_nodes_from(gdf[fid_column].unique())
    if not sjoined.empty:
        sjoined = sjoined[sjoined.index != sjoined[fid_column + '_r']]
        G.add_edges_from(list(zip(sjoined.index, sjoined[fid_column + '_r'])))
    groups = list(nx.connected_components(G))
    if not groups:
        return gdf
    fid_to_gid = {fid: i for i, g in enumerate(groups) for fid in g}
    gdf['_merge_group_id'] = gdf[fid_column].map(fid_to_gid)
    max_gid = gdf['_merge_group_id'].max()
    if pd.isna(max_gid):
        max_gid = -1
    unmapped = gdf['_merge_group_id'].isna()
    n_unmapped = int(unmapped.sum())
    if n_unmapped > 0:
        gdf.loc[unmapped, '_merge_group_id'] = range(int(max_gid) + 1, int(max_gid) + 1 + n_unmapped)
    gdf['_merge_group_id'] = gdf['_merge_group_id'].astype(int)

    agg = {height_column: 'max', form_id_column: 'first'}
    other = [c for c in gdf.columns if c not in (fid_column, height_column, form_id_column, 'geometry', '_merge_group_id')]
    for c in other:
        agg[c] = 'first'
    merged = gdf.dissolve(by='_merge_group_id', aggfunc=agg).reset_index(drop=True)
    print(f"  合并前: {original_count}  →  合并后: {len(merged)}（合并了 {original_count - len(merged)} 个）")
    return merged


def clean_building_data(input_shp, output_shp=None, cfg=None):
    """清洗建筑轮廓；返回清洗后的 GeoDataFrame，若给 output_shp 则写出。"""
    cfg = cfg or {}
    enable_merge = cfg.get('enable_merge', True)
    min_area = float(cfg.get('min_area', 30.0))
    max_elongation = float(cfg.get('max_elongation', 15.0))
    elongation_area_limit = float(cfg.get('elongation_area_limit', 500.0))
    buffer_distance = float(cfg.get('merge_buffer_distance', 0.001))
    fid_column = cfg.get('fid_column', 'form_id')
    height_column = cfg.get('height_column', 'Height')
    form_id_column = cfg.get('form_id_column', 'form_id')
    compute_crs = cfg.get('compute_crs', 'EPSG:4547')

    print(f"\n{'=' * 60}\n建筑数据清洗（融合版）\n{'=' * 60}")
    print(f"输入: {input_shp} | 最小面积={min_area}m² | 最大伸长度={max_elongation}(仅面积<{elongation_area_limit}m²) | 合并={'开' if enable_merge else '关'}")

    gdf = gpd.read_file(input_shp)
    initial = len(gdf)
    print(f"原始要素: {initial}")

    if fid_column not in gdf.columns:
        detected = None
        for c in ['FID', 'TARGET_FID', 'fid', 'Id', 'ID', 'OBJECTID', 'objectid', 'id']:
            if c in gdf.columns:
                detected = c
                break
        if detected is None:
            gdf['_temp_index'] = gdf.index
            fid_column = '_temp_index'
        else:
            fid_column = detected
    if form_id_column not in gdf.columns:
        gdf[form_id_column] = gdf[fid_column]
    gdf[form_id_column] = gdf[form_id_column].astype(gdf[fid_column].dtype)

    original_crs = gdf.crs
    gdf_proj = gdf.to_crs(compute_crs) if (gdf.crs is not None and gdf.crs.is_geographic) else gdf.copy()

    if enable_merge:
        gdf_proj = merge_overlapping_buildings(gdf_proj, fid_column, height_column, form_id_column, buffer_distance)

    print("\n[清洗步骤 1/3] 几何有效性 + 类型检查...")
    gdf_proj['geometry'] = gdf_proj.geometry.buffer(0)
    gdf_proj = gdf_proj[gdf_proj.geometry.is_valid & ~gdf_proj.geometry.is_empty]
    gdf_proj = gdf_proj[gdf_proj.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    print(f"  通过几何检查: {len(gdf_proj)}")

    print(f"\n[清洗步骤 2/3] 面积阈值 (< {min_area}m² 剔除)...")
    if len(gdf_proj) > 0:
        gdf_proj['_area'] = gdf_proj.geometry.area
        gdf_proj = gdf_proj[gdf_proj['_area'] >= min_area]
        print(f"  通过面积检查: {len(gdf_proj)}")

    print(f"\n[清洗步骤 3/3] 伸长度(线状)过滤...")
    if len(gdf_proj) > 0:
        gdf_proj['_elong'] = gdf_proj.geometry.apply(calculate_elongation)
        fail = (gdf_proj['_elong'] > max_elongation) & (gdf_proj['_area'] < elongation_area_limit)
        gdf_proj = gdf_proj[~fail].drop(columns=['_area', '_elong'])
        print(f"  通过伸长度检查: {len(gdf_proj)}")

    if original_crs is not None and str(gdf_proj.crs) != str(original_crs):
        gdf_proj = gdf_proj.to_crs(original_crs)

    print(f"\n清洗统计: 原始 {initial} → 最终 {len(gdf_proj)}（剔除/合并 {initial - len(gdf_proj)}）")
    if output_shp and len(gdf_proj) > 0:
        os.makedirs(os.path.dirname(output_shp), exist_ok=True)
        out = gdf_proj.drop(columns=['_temp_index'], errors='ignore')
        out.to_file(output_shp, driver='ESRI Shapefile', encoding='utf-8')
        print(f"[完成] 清洗结果已写出: {output_shp}")
    return gdf_proj


class UrbanMorphologyGraphBuilder:
    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # 高度开关：data.has_height=false 时数据无高程，自动剔除 Height 特征
        self.has_height = bool(config['data'].get('has_height', True))
        # 基础特征模板（不含 Height）；Height 位于 Covered_area_ratio 之后(索引5)
        self.feature_names = [
            'Area', 'Perimeter', 'Longest_chord_length', 'Mean_radius',
            'Covered_area_ratio',
            'SBRO', 'LCO', 'Bisector_orientation', 'Weighted_orientation',
            'Complexity_index', 'IPQ', 'Fractality', 'Max_circularity',
            'Gibbs_compactness', 'Elongation_index', 'Ellipticity', 'Solidity',
            'DCM', 'Exchange_index', 'BCI',
            'Cross_moment_mu11', 'Variance_spreading',
        ]
        if self.has_height:
            self.feature_names.insert(5, 'Height')  # 有高程时在第6位(索引5)插回 Height，与历史 23 维完全一致
        self.n_features = len(self.feature_names)    # 23(有高度) 或 22(无高度)

    def load_and_preprocess_data(self):
        print('正在加载数据...')
        clean_cfg = self.config.get('cleaning', {})
        if clean_cfg.get('enable', False):
            raw = clean_cfg.get('raw_buildings_shp', self.config['data']['buildings_shp'])
            cleaned = clean_cfg.get('cleaned_buildings_shp', self.config['data']['buildings_shp'])
            print(f'[清洗] 启用：{raw} → 清洗 → {cleaned}')
            buildings = clean_building_data(raw, cleaned, cfg=clean_cfg)
            if buildings is None or len(buildings) == 0:
                raise RuntimeError('清洗后建筑数据为空，请检查 cleaning 参数与输入数据。')
        else:
            buildings = gpd.read_file(self.config['data']['buildings_shp'])
        landuse = gpd.read_file(self.config['data']['landuse_shp'])
        print(f'建筑数据: {len(buildings)} 条  |  土地利用数据: {len(landuse)} 条')

        target_crs = self.config['data'].get('crs', 'EPSG:4547')
        if str(buildings.crs) != target_crs:
            buildings = buildings.to_crs(target_crs)
        if str(landuse.crs) != target_crs:
            landuse = landuse.to_crs(target_crs)

        buildings['centroid_x'] = buildings.geometry.centroid.x
        buildings['centroid_y'] = buildings.geometry.centroid.y
        print(f'坐标系已统一为: {target_crs}')
        return buildings, landuse

    def _detect_building_id_field(self, buildings):
        preferred = self.config['data'].get('buildings_id_field', None)
        candidates = []
        if preferred:
            candidates.append(preferred)
        candidates.extend(['TARGET_FID', 'FID', 'OBJECTID', 'Id', 'ID', 'fid', 'id'])
        for col in candidates:
            if col in buildings.columns:
                return col
        return None

    def compute_morphological_features(self, geometry, building_height, voronoi_area, block_geom):
        coords = _get_polygon_coords(geometry)
        if len(coords) < 3:
            return [0.0] * self.n_features

        centroid = geometry.centroid
        cx, cy = centroid.x, centroid.y
        area = float(geometry.area)
        perimeter = float(geometry.length)
        ch_coords, convex_hull = _get_convex_hull_coords(geometry)
        ch_area = float(convex_hull.area) if convex_hull.area > 0 else 1.0
        distances = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
        mean_radius = float(np.mean(distances))
        l_max, lco_angle, chord_indices = _compute_longest_chord(ch_coords)
        l_sbr, w_sbr, sbro_angle = _compute_minimum_bounding_rectangle(geometry)

        f_covered_ratio = area / voronoi_area if voronoi_area > 0 else 1.0
        f_covered_ratio = min(f_covered_ratio, 1.0)
        f_height = float(building_height) if pd.notnull(building_height) else 0.0

        if area > 0 and perimeter > 1:
            f_fractality = 1.0 - np.log(area) / (2.0 * np.log(perimeter))
        else:
            f_fractality = 0.0

        # 新增：论文 Table 1 中补全的 2 个 shape 指标
        # Cross_moment_mu11 (μ11)：顶点相对质心的二阶交叉矩 Σ(x_i-cx)(y_i-cy)，刻画旋转不对称性（非完整惯性张量）
        f_cross_moment = _compute_cross_moment(coords, cx, cy)
        # Variance (spreading)：顶点相对质心在 x、y 方向方差之和 [Σ(x-cx)²+Σ(y-cy)²]/n
        f_variance_spreading = _compute_variance_spreading(coords, cx, cy)

        feats = [
            area,
            perimeter,
            l_max,
            mean_radius,
            f_covered_ratio,
            f_height,
            sbro_angle,
            lco_angle,
            _compute_bisector_orientation(ch_coords, chord_indices, lco_angle),
            _compute_weighted_orientation(coords),
            perimeter / area if area > 0 else 0.0,  # P/A (perimeter-area ratio)，FRAGSTATS P1 类 shape complexity 代理
            (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0,
            f_fractality,
            _compute_max_circularity(area, distances),
            (4.0 * area) / (np.pi * l_max ** 2) if l_max > 0 else 0.0,
            l_sbr / w_sbr if w_sbr > 0 else 1.0,
            _compute_ellipticity(coords, lco_angle, l_max),
            area / ch_area if ch_area > 0 else 1.0,
            _compute_dcm(coords, area),
            _compute_exchange_index(area, geometry),
            _compute_bci(distances),
            f_cross_moment,
            f_variance_spreading,
        ]
        if not self.has_height:
            # 无高程：剔除第6个(索引5) Height 特征，与 self.feature_names 对齐（其余特征位置不变）
            feats = feats[:5] + feats[6:]
        return feats

    def merge_close_buildings(self, buildings, threshold=1.0, id_field=None):
        if len(buildings) < 2 or threshold <= 0:
            if id_field and 'building_ids_group' not in buildings.columns:
                buildings = buildings.copy()
                buildings['building_ids_group'] = buildings[id_field].astype(str).apply(lambda x: [x])
            return buildings

        buildings = buildings.copy()
        if id_field and 'building_ids_group' not in buildings.columns:
            buildings['building_ids_group'] = buildings[id_field].astype(str).apply(lambda x: [x])

        centroids = buildings[['centroid_x', 'centroid_y']].to_numpy(dtype=float)
        kdtree = KDTree(centroids)
        pairs = kdtree.query_pairs(threshold)
        if not pairs:
            return buildings

        n = len(buildings)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i, j in pairs:
            union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        height_field = self.config['data'].get('buildings_height_field', 'Height')
        merged_rows = []
        for _, indices in groups.items():
            if len(indices) == 1:
                row = buildings.iloc[indices[0]].to_dict()
                merged_rows.append(row)
            else:
                group_b = buildings.iloc[indices]
                merged_geom = unary_union(list(group_b.geometry))
                base = group_b.iloc[0].copy()
                base['geometry'] = merged_geom
                base['centroid_x'] = merged_geom.centroid.x
                base['centroid_y'] = merged_geom.centroid.y
                if self.has_height and height_field in base.index:
                    heights = pd.to_numeric(group_b[height_field], errors='coerce').dropna()
                    base[height_field] = heights.max() if len(heights) > 0 else 0.0
                if 'building_ids_group' in group_b.columns:
                    merged_ids = []
                    for item in group_b['building_ids_group'].tolist():
                        if isinstance(item, list):
                            merged_ids.extend([str(x) for x in item])
                        else:
                            merged_ids.append(str(item))
                    base['building_ids_group'] = merged_ids
                merged_rows.append(base.to_dict())
        return gpd.GeoDataFrame(merged_rows, crs=buildings.crs)

    def _delaunay_edges(self, centroids):
        """返回 (edges, dists)：Delaunay 三角剖分得到的去重无向边及其长度。"""
        try:
            tri = Delaunay(centroids, qhull_options='QJ')
            simplices = tri.simplices
            if len(simplices) == 0:
                return None, None
            edges = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
            edges = np.sort(edges, axis=1)
            edges = np.unique(edges, axis=0)
            diffs = centroids[edges[:, 0]] - centroids[edges[:, 1]]
            dists = np.linalg.norm(diffs, axis=1)
            return edges, dists
        except Exception:
            return None, None

    def _build_edges_by_delaunay(self, centroids, dist_thresh):
        edges, dists = self._delaunay_edges(centroids)
        if edges is None:
            return None
        edges = edges[dists <= dist_thresh]   # [P2] 按配置/自动的距离阈值剪边（依据南京城墙论文 200m）
        if len(edges) == 0:
            return None
        rev_edges = edges[:, ::-1]
        return np.vstack([edges, rev_edges]).T.astype(np.int64)

    def _collect_all_edge_lengths(self, grouped_buildings, block_to_label, merge_thresh, min_b, building_id_field):
        """[auto] 自动化阈值用：遍历各地块收集 Delaunay 边长分布（看分布、选截断阈值）。
        注：为效率使用 merge 后的全部建筑质心（与构图主体一致；少数特征计算失败的建筑不影响阈值估计）。"""
        all_lengths = []
        for block_id, b_in_block in tqdm(grouped_buildings.items(), desc='收集边长',
                                          disable=not sys.stdout.isatty()):
            if block_id not in block_to_label:
                continue
            if len(b_in_block) < min_b:
                continue
            b_in_block = self.merge_close_buildings(b_in_block, merge_thresh, building_id_field)
            if len(b_in_block) < min_b:
                continue
            centroids = b_in_block[['centroid_x', 'centroid_y']].to_numpy(dtype=float)
            _, dists = self._delaunay_edges(centroids)
            if dists is not None and len(dists) > 0:
                all_lengths.append(dists)
        if not all_lengths:
            return np.array([])
        return np.concatenate(all_lengths)

    def process_all_blocks(self, buildings, landuse):
        print('\n正在构建城市形态图...')
        b_block_col = self.config['data']['buildings_block_id_field']
        l_block_col = self.config['data']['landuse_block_id_field']
        label_col = self.config['data']['landuse_label_field']
        height_field = self.config['data'].get('buildings_height_field', 'Height')
        merge_thresh = self.config['graph'].get('centroid_merge_threshold', 1.0)
        fixed_thresh = self.config['graph'].get('distance_threshold', 200)  # [P2] 默认 200m
        min_b = self.config['graph'].get('min_buildings_per_block', 4)
        building_id_field = self._detect_building_id_field(buildings)
        print(f'检测到建筑ID字段: {building_id_field if building_id_field else "未找到，将退化为地块内顺序ID"}')

        if building_id_field is None:
            buildings = buildings.copy()
            buildings['__temp_building_id__'] = buildings.index.astype(str)
            building_id_field = '__temp_building_id__'

        landuse_labels = landuse[label_col].astype(str)
        landuse['encoded_label'] = self.label_encoder.fit_transform(landuse_labels)
        print(f'标签编码: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}')

        block_to_label = dict(zip(landuse[l_block_col], landuse['encoded_label']))
        block_to_geom = dict(zip(landuse[l_block_col], landuse.geometry))
        grouped_buildings = {bid: grp.copy() for bid, grp in buildings.groupby(b_block_col, sort=False)}
        all_blocks = list(grouped_buildings.keys())

        # [auto] 边阈值自动化：先收集全部 Delaunay 边长看分布，再按分位数截断
        threshold_mode = self.config['graph'].get('threshold_mode', 'fixed')
        auto_quantile = self.config['graph'].get('auto_quantile', 0.95)
        dist_thresh = fixed_thresh
        self.edge_lengths = None
        self.chosen_threshold = fixed_thresh
        self.threshold_mode = threshold_mode
        if threshold_mode == 'auto':
            print('\n[auto] 边阈值自动化：先收集全部 Delaunay 边长看分布...')
            self.edge_lengths = self._collect_all_edge_lengths(
                grouped_buildings, block_to_label, merge_thresh, min_b, building_id_field)
            if self.edge_lengths is None or len(self.edge_lengths) == 0:
                print('[auto] 未收集到任何边长，回退使用固定 distance_threshold =', fixed_thresh)
            else:
                dist_thresh = float(np.quantile(self.edge_lengths, auto_quantile))
                self.chosen_threshold = dist_thresh
                print(f'[auto] 边长分布 n={len(self.edge_lengths)} | '
                      f'min={self.edge_lengths.min():.1f} p50={np.quantile(self.edge_lengths,0.5):.1f} '
                      f'p90={np.quantile(self.edge_lengths,0.9):.1f} p95={np.quantile(self.edge_lengths,0.95):.1f} '
                      f'p99={np.quantile(self.edge_lengths,0.99):.1f} max={self.edge_lengths.max():.1f}')
                print(f'[auto] 选定阈值 = 第 {auto_quantile*100:.0f} 分位数 = {dist_thresh:.1f} m '
                      f'(对应"尾部密度近零"；论文 200m 为其 empiric 落点)')
        print(f'[P2] 距离阈值 distance_threshold = {dist_thresh} m  (mode={threshold_mode})')

        raw_graphs = []
        block_stats = []

        for block_id in tqdm(all_blocks, desc='构建图',
                              disable=not sys.stdout.isatty()):
            if block_id not in block_to_label:
                continue
            b_in_block = grouped_buildings[block_id]
            if len(b_in_block) < min_b:
                continue
            b_in_block = self.merge_close_buildings(b_in_block, merge_thresh, building_id_field)
            if len(b_in_block) < min_b:
                continue

            label = int(block_to_label[block_id])
            block_geom = block_to_geom.get(block_id)
            centroids = b_in_block[['centroid_x', 'centroid_y']].to_numpy(dtype=float)
            voronoi_areas = _compute_voronoi_areas(centroids, block_geom)

            features_list = []
            valid_centroids = []
            valid_building_ids_group = []
            valid_building_ids_str = []

            for idx, row in enumerate(b_in_block.itertuples(index=False)):
                try:
                    geom = row.geometry
                    h = getattr(row, height_field, 0.0) if (self.has_height and hasattr(row, height_field)) else 0.0
                    feats = self.compute_morphological_features(geom, h, voronoi_areas[idx] if idx < len(voronoi_areas) else 1.0, block_geom)
                    features_list.append(feats)
                    valid_centroids.append([row.centroid_x, row.centroid_y])
                    if hasattr(row, 'building_ids_group'):
                        ids_group = getattr(row, 'building_ids_group')
                        if not isinstance(ids_group, list):
                            ids_group = [str(ids_group)]
                    else:
                        ids_group = [str(getattr(row, building_id_field))]
                    valid_building_ids_group.append(ids_group)
                    valid_building_ids_str.append('|'.join([str(x) for x in ids_group]))
                except Exception:
                    continue

            if len(features_list) < min_b:
                continue

            X = np.array(features_list, dtype=np.float32)
            centroids = np.array(valid_centroids, dtype=float)
            edge_index = self._build_edges_by_delaunay(centroids, dist_thresh)
            if edge_index is None:
                continue

            raw_graphs.append({
                'x': X,
                'edge_index': edge_index,
                'y': label,
                'block_id': block_id,
                'centroids': centroids,
                'building_ids_group': valid_building_ids_group,
                'building_ids_str': valid_building_ids_str,
            })

            block_stats.append({
                'block_id': block_id,
                'num_nodes': len(X),
                'num_edges': edge_index.shape[1] // 2,
                'label': label,
                'label_name': self.label_encoder.inverse_transform([label])[0],
            })

        print(f'\n成功构建 {len(raw_graphs)} / {len(all_blocks)} 个地块的图')
        if not raw_graphs:
            return [], [], building_id_field

        print('正在标准化特征（全局 StandardScaler）...')
        all_X = np.vstack([g['x'] for g in raw_graphs])
        self.feature_scaler.fit(all_X)

        pyg_dataset = []
        for g in raw_graphs:
            x_scaled = self.feature_scaler.transform(g['x'])
            x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            data = Data(
                x=torch.tensor(x_scaled, dtype=torch.float),
                edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
                y=torch.tensor([g['y']], dtype=torch.long),
                block_id=g['block_id'],
                num_nodes=len(x_scaled),
            )
            # [P1] 把建筑质心一并存入 Data，供训练阶段按真实地理距离实时算边权
            data.centroids = torch.tensor(g['centroids'], dtype=torch.float)
            data.building_ids_str = g['building_ids_str']
            data.building_ids_group = g['building_ids_group']
            pyg_dataset.append(data)

        return pyg_dataset, block_stats, building_id_field


def main():
    print('=' * 70)
    print('城市形态图构建器（优化实验版 opt）| 距离边权质心 + 自动检查导出')
    print('=' * 70)

    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'   # [P0] 可传自定义配置文件（如 config_auto_thresh.yaml）
    if not os.path.exists(config_path):
        print(f'错误: 未找到配置文件 {config_path}')
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _run_dir = os.environ.get('PIPELINE_RUN_DIR')
    base = _run_dir if _run_dir else config['output']['base_output_dir']
    out_dir = os.path.join(base, '01_Graphs', f'graphs_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    builder = UrbanMorphologyGraphBuilder(config)
    buildings, landuse = builder.load_and_preprocess_data()
    dataset, stats, building_id_field = builder.process_all_blocks(buildings, landuse)

    if not dataset:
        print('错误: 未能构建任何图！请检查数据和配置。')
        return

    torch.save(dataset, os.path.join(out_dir, 'graph_dataset.pt'))
    metadata = {
        'label_encoder': builder.label_encoder,
        'feature_scaler': builder.feature_scaler,
        'feature_names': builder.feature_names,
        'building_id_field': building_id_field,
        'config': config,
    }
    with open(os.path.join(out_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    pd.DataFrame(stats).to_csv(os.path.join(out_dir, 'block_stats.csv'), index=False, encoding='utf-8-sig')
    export_graph_examination(dataset, out_dir, builder.feature_names, builder.label_encoder)

    # [auto] 若本次为自动阈值，保存边长分布图与统计（供人工查看分布、手动微调）
    if getattr(builder, 'edge_lengths', None) is not None and getattr(builder, 'threshold_mode', None) == 'auto':
        try:
            lengths = builder.edge_lengths
            counts, edges_hist = np.histogram(lengths, bins=100)
            info = {
                'threshold_mode': 'auto',
                'auto_quantile': config['graph'].get('auto_quantile', 0.95),
                'chosen_threshold_m': float(builder.chosen_threshold),
                'n_edges': int(len(lengths)),
                'min': float(lengths.min()), 'p50': float(np.quantile(lengths, 0.5)),
                'p90': float(np.quantile(lengths, 0.9)), 'p95': float(np.quantile(lengths, 0.95)),
                'p99': float(np.quantile(lengths, 0.99)), 'max': float(lengths.max()),
                'histogram': {'bin_edges': edges_hist.tolist(), 'counts': counts.tolist()},
            }
            with open(os.path.join(out_dir, 'edge_threshold.json'), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as _plt
                _plt.figure(figsize=(10, 5))
                _plt.hist(lengths, bins=100, color='steelblue', edgecolor='white')
                _plt.axvline(builder.chosen_threshold, color='red', linestyle='--',
                             label=f'选定阈值 {builder.chosen_threshold:.1f}m')
                _plt.xlabel('Delaunay 边长 (m)'); _plt.ylabel('边数')
                _plt.title('自动边阈值：Delaunay 边长分布')
                _plt.legend(); _plt.tight_layout()
                _plt.savefig(os.path.join(out_dir, 'edge_length_distribution.png'), dpi=150)
                _plt.close()
                print(f'   已保存边长分布图: {os.path.join(out_dir, "edge_length_distribution.png")}')
            except Exception as e:
                print(f'   (跳过分布图绘制: {e})')
            print(f'   已保存自动阈值统计: {os.path.join(out_dir, "edge_threshold.json")}')
        except Exception as e:
            print(f'   [auto] 保存边长分布失败: {e}')

    print(f'\n[完成] 成功构建 {len(dataset)} 个图！')
    print(f'   图数据已保存至: {out_dir}')
    print(f'   特征数量: {len(builder.feature_names)}')
    print(f'   建筑ID字段: {building_id_field}')
    print('   已自动导出: examine.csv / feature_summary.csv / graph_summary.csv / examine_run_info.json')

    # [P0] 自动把本次输出的图目录写回【实际加载的配置文件】，修复旧版接链 bug
    # 注意：用「逐行替换」而非 yaml.dump 重写，保留用户写在 config.yaml 里的全部注释。
    train_cfg_path = config_path
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, 'r', encoding='utf-8') as f:
            cfg_lines = f.readlines()
        new_lines = []
        for line in cfg_lines:
            stripped = line.strip()
            # 更新 training_paths.graph_data_dir
            if stripped.startswith('graph_data_dir:'):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}graph_data_dir: {out_dir}\n")
                continue
            # 仅 auto 模式更新 graph.distance_threshold
            if getattr(builder, 'threshold_mode', None) == 'auto' and stripped.startswith('distance_threshold:'):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}distance_threshold: {float(builder.chosen_threshold)}\n")
                continue
            new_lines.append(line)
        with open(train_cfg_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f'   已自动更新 {train_cfg_path} 中的 graph_data_dir -> {out_dir}')

    print('\n下一步: python -m src.GNN_Train（③ 训练）；或 python -m pipelines.run_pipeline 串起全流程')


if __name__ == '__main__':
    main()
