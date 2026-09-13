# =============================================================================
# prepare_city.py  —— 跨城市「实验前准备」本地脚本（无联网下载）
# -----------------------------------------------------------------------------
# 本脚本只做本地处理，不联网、不自动下载。用户自己准备好两类数据：
#   形态数据：某城市的 SinoBF-1 建筑轮廓（.shp 或 .zip），来自 Zenodo 17844789
#            命名形如 G_P_C.zip（如 East_Shanghai_SinoBF-1.zip），解压后 <stem>/<stem>.shp
#   功能数据：全国 EULUC-China 2.0（.shp 或 .zip），来自 Zenodo 16794007
#            解压后含 EULUC_China_20_1.shp / _2.shp 等多片
#
# 处理流程（A1：保留映射，不做下载）：
#   Step 1  读建筑轮廓 -> 清洗（剔空几何 + buffer(0) 修无效几何）
#   Step 2  读全国 EULUC -> 按 bbox 裁出该城市功能标签
#           -> Class(0-10) 映射 Level1(1-5) + 生成 block_id        （★映射在这里做）
#   Step 3  建筑质心 within 连 EULUC 地块挂 block_id -> 写出 sinobf_<city>_buildings.shp
#
# 设计原则：
#   - 幂等：本地已存在则跳过（解压/裁都如此），--force 可强制重做解压。
#   - 可控：用 --sinobf-shp/--sinobf-zip/--euluc-shp/--euluc-zip/--func-shp 指定本地文件。
#   - 缺失即报错：任一必需本地文件不存在，直接清晰报错并提示去 README/Zenodo 取，绝不联网。
#
# 用法：
#   # 标准：建筑 zip + 全国 EULUC zip（本机已有），离线跑（在项目根执行）
#   python data_prep/prepare_city.py --city Shanghai \
#       --sinobf-zip data/East_Shanghai_SinoBF-1.zip \
#       --euluc-zip data/EULUC_China_20_SHP.zip
#
#   # 功能标签已自己裁好：跳过 Step 2
#   python data_prep/prepare_city.py --city Shanghai \
#       --sinobf-zip data/East_Shanghai_SinoBF-1.zip \
#       --func-shp data/function_Shanghai.shp
#
# 完成后：把 config.yaml 的 data.buildings_shp / data.landuse_shp 指向产出文件，
#         再 python pipelines/run_pipeline.py
# =============================================================================
import argparse
import os
import sys
import time
import zipfile
import geopandas as gpd
import numpy as np
import pandas as pd

# 让 Windows 子进程 stdout 即使遇到意外字符也不崩（落到 ? 而非中断）
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

EULUC_ZIP_DEFAULT = 'EULUC_China_20_SHP.zip'

# EULUC-China 2.0 的 Class(0-10, level2 细分) -> Level1(1-5, 5 大类) 映射。
# 取自广州 GBA function.shp 的 (Class, Level1) 黄金对照，确定性、可复现。
#   0->1 住宅 | 1,2->2 商业 | 3->3 工业 | 4,5->4 交通 | 6,7,8,9,10->5 公共管理与服务
CLASS_TO_LEVEL1 = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4, 5: 4,
                   6: 5, 7: 5, 8: 5, 9: 5, 10: 5}
LEVEL1_LAB = {1: '住宅', 2: '商业', 3: '工业', 4: '交通', 5: '公共管理与服务'}


# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------
def require_local(path, what):
    """检查本地文件存在；不存在则清晰报错（不联网）。"""
    if path and os.path.exists(path):
        return path
    raise SystemExit(
        f"[错误] 找不到{what}：{path}\n"
        f"  请先用浏览器从 Zenodo 下好对应数据并放到本地，再运行本脚本。\n"
        f"  数据来源与步骤见 README.md 的「数据准备」一节。")


def unzip_if_needed(zip_path, extract_dir, force=False):
    if os.path.isdir(extract_dir) and os.listdir(extract_dir) and not force:
        print(f'  [跳过] 已解压: {extract_dir}')
        return
    print(f'  解压 {zip_path} -> {extract_dir} ...')
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    print('  解压完成')


def find_main_shp(directory, contains=None):
    candidates = []
    for root, _, files in os.walk(directory):
        for f in files:
            fl = f.lower()
            if fl.endswith('.shp') and not fl.endswith('.shp.xml'):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise SystemExit(f'[错误] 在 {directory} 未找到 .shp 文件')
    if contains:
        filtered = [c for c in candidates if contains.lower() in os.path.basename(c).lower()]
        if filtered:
            candidates = filtered
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


def sinobf_shp_default(zip_name):
    stem = zip_name[:-4]
    return os.path.join('data', stem, stem + '.shp')


def euluc_unzip_dir_default():
    return os.path.join('data', EULUC_ZIP_DEFAULT[:-4])


# ----------------------------------------------------------------------------
# Step 1: 定位 + 清洗建筑形态数据（仅本地）
# ----------------------------------------------------------------------------
def locate_sinobf_shp(args):
    """定位城市建筑形态 shp（必须本机已有，不联网）。"""
    if args.sinobf_shp:
        require_local(args.sinobf_shp, '建筑形态 shp (--sinobf-shp)')
        print(f'[Step 1] 本地建筑形态 shp: {args.sinobf_shp}')
        return args.sinobf_shp
    if args.sinobf_zip:
        require_local(args.sinobf_zip, '建筑形态 zip (--sinobf-zip)')
        zip_name = os.path.basename(args.sinobf_zip)
        print(f'[Step 1] 本地 SinoBF zip: {args.sinobf_zip}')
        extract_dir = os.path.join('data', zip_name[:-4])
        unzip_if_needed(args.sinobf_zip, extract_dir, force=args.force)
        shp = sinobf_shp_default(zip_name)
        if not os.path.exists(shp):
            shp = find_main_shp(extract_dir)
        return shp
    raise SystemExit(
        '[错误] 未提供建筑形态数据。请用 --sinobf-shp <本地shp> 或 '
        '--sinobf-zip <本地zip> 指定（数据需自己从 Zenodo 下好）。')


def load_clean_buildings(shp_path):
    """读取建筑轮廓并清洗：剔除空几何 + buffer(0) 修复无效几何。返回 (gdf, 原始条数)"""
    t0 = time.time()
    print(f'[Step 1] 加载并清洗建筑轮廓: {shp_path}')
    bld = gpd.read_file(shp_path)
    n0 = len(bld)
    print(f'    原始建筑数: {n0} | CRS: {bld.crs}')

    bld = bld[~bld.geometry.isnull()]
    bld = bld[~bld.geometry.is_empty]
    bad = ~bld.geometry.is_valid
    if bad.any():
        n_bad = int(bad.sum())
        bld.loc[bad, 'geometry'] = bld.loc[bad, 'geometry'].buffer(0)
        bld = bld[~bld.geometry.is_empty]
        print(f'    修复无效几何: {n_bad} 个 -> buffer(0)')
    print(f'    清洗后建筑数: {len(bld)} | 耗时 {time.time() - t0:.1f}s')
    return bld, n0


def bbox_of_gdf(gdf, pad_deg):
    """由建筑自身范围推出 bbox（EPSG:4326 度），外扩 pad_deg 度避免边缘地块被切掉。"""
    g = gdf if (gdf.crs is not None and gdf.crs.to_epsg() == 4326) else gdf.to_crs('EPSG:4326')
    minx, miny, maxx, maxy = g.total_bounds
    return [minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg]


# ----------------------------------------------------------------------------
# Step 2: 裁剪 EULUC 全国 -> 单城市功能标签地块（本地，含 Class->Level1 映射）
# ----------------------------------------------------------------------------
def prepare_euluc(args, city, bbox):
    out_func = args.func_shp or os.path.join('data', f'function_{city}.shp')
    if args.func_shp:
        require_local(args.func_shp, '功能标签地块 (--func-shp)')
        print(f'[Step 2] 使用已提供的功能标签地块，跳过裁剪: {args.func_shp}')
        return out_func

    shp_list = []
    if args.euluc_shp:
        require_local(args.euluc_shp, '全国 EULUC shp (--euluc-shp)')
        shp_list = [args.euluc_shp]
    else:
        euluc_zip = args.euluc_zip or os.path.join('data', EULUC_ZIP_DEFAULT)
        require_local(euluc_zip, '全国 EULUC zip (--euluc-zip 或放到 data/EULUC_China_20_SHP.zip)')
        extract_dir = euluc_unzip_dir_default()
        unzip_if_needed(euluc_zip, extract_dir, force=args.force)
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith('.shp') and 'euluc' in f.lower():
                    shp_list.append(os.path.join(root, f))
        if not shp_list:
            raise SystemExit(f'[错误] 在 {extract_dir} 未找到 EULUC shp 文件')

    print(f'[Step 2] 从全国 EULUC 2.0 按 bbox 空间过滤裁剪 {city} ...')
    print(f'    bbox (EPSG:4326): {bbox}')
    print(f'    读入分片数: {len(shp_list)}')
    parts = []
    for fp in shp_list:
        parts.append(gpd.read_file(fp, bbox=tuple(bbox)))
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    print(f'    裁出要素数: {len(gdf)}')
    if len(gdf) == 0:
        raise SystemExit('[错误] 裁出 0 要素，检查 bbox 与全国 shp 坐标系（应均为 EPSG:4326）')

    # 全国 EULUC 原始仅含 Class + geometry，需补齐与本流水线一致的字段：
    #   block_id   : 地块唯一 ID（顺序编号，供建筑 spatial join 挂接）
    #   Level1     : 5 大类标签(float，与广州 function.shp 同构，保证跨城市编码器一致)
    #   Level1_lab : 大类中文名
    gdf = gdf.reset_index(drop=True)
    gdf['block_id'] = gdf.index + 1
    gdf['Level1'] = gdf['Class'].map(CLASS_TO_LEVEL1).astype(float)
    if gdf['Level1'].isna().any():
        miss = sorted(set(gdf['Class'].dropna().astype(int).tolist()) -
                      set(CLASS_TO_LEVEL1.keys()))
        raise SystemExit(f'[错误] 出现未知 Class 值未映射: {miss}。请更新 CLASS_TO_LEVEL1。')
    gdf['Level1_lab'] = gdf['Level1'].map(LEVEL1_LAB)
    gdf.to_file(out_func)
    print(f'    写出功能标签地块: {out_func} (cols: {list(gdf.columns)})')
    return out_func


# ----------------------------------------------------------------------------
# Step 3: 清洗 + 接入 SinoBF 建筑
# ----------------------------------------------------------------------------
def prepare_sinobf(args, city, func_shp, bld=None, n0=None):
    out_build = args.out_buildings or os.path.join('data', f'sinobf_{city}_buildings.shp')

    if bld is None:
        sinobf_shp = locate_sinobf_shp(args)
        bld, n0 = load_clean_buildings(sinobf_shp)

    t0 = time.time()
    print(f'[Step 3] 挂接建筑到地块 ({city}) ...')
    sinobf = bld

    # 统一坐标系到 EULUC
    func = gpd.read_file(func_shp)[['block_id', 'geometry']].copy()
    if str(sinobf.crs) != str(func.crs):
        print(f'    重投影 SinoBF: {sinobf.crs} -> {func.crs}')
        sinobf = sinobf.to_crs(func.crs)
    else:
        print(f'    坐标系已一致: {func.crs}')
    print(f'    EULUC 地块数: {len(func)}')

    # 以建筑质心做 within 连接到 EULUC 地块，挂 block_id
    print('    以建筑质心 within 连 EULUC 地块（落不到地块的建筑剔除）...')
    centroids = gpd.GeoSeries(sinobf.geometry.centroid, crs=sinobf.crs)
    pts = gpd.GeoDataFrame(geometry=centroids, crs=sinobf.crs)
    joined = gpd.sjoin(pts, func, how='inner', predicate='within')
    joined = joined[~joined.index.duplicated(keep='first')]
    mapping = dict(zip(joined.index, joined['block_id']))
    sinobf['block_id'] = [mapping.get(i) for i in sinobf.index]
    kept = sinobf[sinobf['block_id'].notna()].copy()
    kept['block_id'] = kept['block_id'].astype(int)
    dropped = n0 - len(kept)
    print(f'    保留: {len(kept)} | 剔除(质心不在任何地块): {dropped} '
          f'(占比 {dropped / n0 * 100:.1f}%)')

    print('    写出...')
    kept = kept.copy()
    kept['buildingid'] = np.arange(len(kept), dtype=int)
    out = kept[['buildingid', 'block_id', 'geometry']].copy()
    out.to_file(out_build)
    print(f'    写出建筑接入文件: {out_build}')
    print(f'    [Step 3] 耗时: {time.time() - t0:.1f}s')
    return out_build


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Cross-city pre-experiment prepare (LOCAL only: clip EULUC + clean & link '
                    'SinoBF buildings). No network download.')
    p.add_argument('--city', default='shanghai',
                   help='城市名（仅用于输出文件命名：function_<city>.shp / sinobf_<city>_buildings.shp）')
    p.add_argument('--force', action='store_true', help='强制重新解压（即使本地已解压）')
    p.add_argument('--bbox', nargs=4, type=float, default=None,
                   metavar=('MINX', 'MINY', 'MAXX', 'MAXY'),
                   help='自定义裁剪范围(EPSG:4326 度)；不给则按建筑范围自动推导')
    p.add_argument('--bbox-pad', type=float, default=0.02,
                   help='自动推导 bbox 时的外扩度数（默认 0.02 度，约 2km）')
    # 本地文件（必须本机已有，不联网下载）
    p.add_argument('--sinobf-shp', default=None, help='本地 SinoBF 城市建筑 shp')
    p.add_argument('--sinobf-zip', default=None, help='本地 SinoBF 城市 zip（需解压）')
    p.add_argument('--euluc-shp', default=None, help='本地全国 EULUC shp（跳过解压+查找）')
    p.add_argument('--euluc-zip', default=None, help='本地全国 EULUC zip')
    p.add_argument('--func-shp', default=None, help='已裁好的城市功能标签 shp（跳过 Step 2 裁剪）')
    p.add_argument('--out-buildings', default=None, help='建筑接入输出 shp（默认 data/sinobf_<city>_buildings.shp）')
    return p.parse_args()


def main():
    args = parse_args()
    city = args.city
    t0 = time.time()
    print('=' * 64)
    print(f'  实验前准备（本地模式）：城市 {city}')
    print('=' * 64)

    # Step 1：定位 + 清洗建筑形态数据
    bld_shp = locate_sinobf_shp(args)
    bld, n0 = load_clean_buildings(bld_shp)

    # Step 1.5：确定裁剪 bbox（--bbox 自定义 > 建筑范围自动推导）
    if args.bbox:
        bbox, bbox_src = list(args.bbox), '--bbox 自定义'
    else:
        bbox = bbox_of_gdf(bld, args.bbox_pad)
        bbox_src = f'建筑范围自动推导(外扩 {args.bbox_pad} 度)'
    print(f'    bbox 来源: {bbox_src}')
    print(f'    bbox (EPSG:4326): {[round(v, 4) for v in bbox]}')

    # Step 2：EULUC 按 bbox 裁出功能标签（含 Class->Level1 映射）
    func_shp = prepare_euluc(args, city, bbox)
    # Step 3：建筑质心挂接到地块
    build_shp = prepare_sinobf(args, city, func_shp, bld=bld, n0=n0)

    print('=' * 64)
    print('  [完成] 实验前准备产出：')
    print(f'    功能标签地块 : {func_shp}')
    print(f'    建筑接入文件 : {build_shp}')
    print(f'    总耗时: {time.time() - t0:.1f}s')
    print()
    print('  下一步：')
    print(f'    1) 让 config.yaml 指向本城产出：')
    print(f'         data.buildings_shp = {build_shp}')
    print(f'         data.landuse_shp   = {func_shp}')
    print(f'    2) 快速验证： python pipelines/run_pipeline.py   （training.quick_fold=true）')
    print(f'    3) 完整 10 折： python pipelines/run_pipeline.py （training.quick_fold=false）')
    print('=' * 64)


if __name__ == '__main__':
    main()
