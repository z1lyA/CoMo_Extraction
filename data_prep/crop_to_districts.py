"""
crop_to_districts.py -- 按行政区边界裁剪建筑 / 地块数据（可选预处理工具）

用途
----
把 config.yaml 里指定的全量建筑 shp 与地块功能 shp，裁到若干行政区范围内，
得到一份更小的数据子集，便于在 CPU 上快速跑通 / 迭代整条 GNN 流水线。

它补的是 bbox 裁剪覆盖不到的场景：prepare_city.py 按经纬度矩形框裁城市，
而本脚本按行政区边界（不规则多边形）精确裁剪。

它不做什么
----------
- 不下载数据（原始数据请先用 prepare_city.py 备好）
- 不建图、不训练，只做几何裁剪
- 默认不覆盖原文件：结果写到新文件名，是否让 config 指过去由你决定

运行方式
--------
    python data_prep/crop_to_districts.py --boundary <区界线.shp> --districts 福田区 南山区

关键参数
--------
--config         配置文件路径，默认项目根的 config.yaml；
                 脚本从中读 data.buildings_shp / data.landuse_shp / data.crs
--boundary       [必填] 行政区界线 shp
--districts      [必填] 要保留的区名，可给多个（如 --districts 福田区 南山区）
--district-field 区界 shp 里的区名字段名；不填则自动探测常见字段名
--list           只列出区界 shp 有哪些区名后退出（用来确认区名怎么写）
--out-dir        输出目录，默认与输入 shp 同目录
--suffix         输出文件名后缀，默认 "_crop"
--in-place       覆盖写回 config 指向的原文件（默认先备份到 temp/）
--no-backup      与 --in-place 连用时不备份（不建议）

示例
----
    # 1) 先看区界文件里都有哪些区
    python data_prep/crop_to_districts.py --boundary path/to/boundary.shp --list

    # 2) 裁两个区，输出到 data/xxx_crop.shp，不动原文件
    python data_prep/crop_to_districts.py --boundary path/to/boundary.shp \
        --districts 福田区 南山区

    # 3) 直接覆盖 config 指向的原文件（自动备份）
    python data_prep/crop_to_districts.py --boundary path/to/boundary.shp \
        --districts 福田区 --in-place

裁剪完成后，把 config.yaml 的 pipeline.run_build 设为 true 重新建图。
"""
import argparse
import glob
import os
import shutil
import sys
from datetime import date

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

import geopandas as gpd

try:
    import yaml
except ImportError:
    print("ERROR: 需要 PyYAML 来读取 config.yaml，请先安装: pip install pyyaml")
    raise SystemExit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# 区名字段的常见命名，按优先级自动探测
_FIELD_CANDIDATES = [
    "行政区名称", "区县名称", "区县", "区名", "行政区", "名称",
    "NAME", "Name", "name", "ADNAME", "adname",
    "DISTRICT", "district", "COUNTY", "county",
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(p):
    """把 config 里的相对路径按项目根解析成绝对路径。"""
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def norm(s):
    return str(s).strip()


def pick_district_field(gdf, districts, explicit):
    """确定区界 shp 里表示区名的字段。"""
    if explicit:
        if explicit not in gdf.columns:
            raise SystemExit(
                "ERROR: 区界 shp 里没有字段 '%s'，可用字段: %s"
                % (explicit, ", ".join(map(str, gdf.columns)))
            )
        return explicit
    names = set()
    if districts:
        names = {norm(d) for d in districts}
    # 1) 优先在常见字段名里找；要求字段值能对上区名（给了区名时）
    for c in _FIELD_CANDIDATES:
        if c not in gdf.columns:
            continue
        if not names:
            return c
        vals = {norm(v) for v in gdf[c].astype(str)}
        if names & vals or any(v in n or n in v for v in vals for n in names):
            return c
    # 2) 兜底：任意字符串型字段，只要值里能找到全部区名
    for c in gdf.columns:
        try:
            if gdf[c].dtype != object:
                continue
        except Exception:
            continue
        vals = {norm(v) for v in gdf[c].astype(str)}
        if names and all(any(n == v or n in v or v in n for v in vals) for n in names):
            return c
    if not names:
        for c in gdf.columns:
            if gdf[c].dtype == object:
                return c
    return None


def match_districts(series, districts):
    """先精确匹配，匹配不到的区再退化为子串匹配（如 '福田' -> '福田区'）。"""
    vals = series.astype(str).map(norm)
    unique = sorted(set(vals))
    hit, missing = [], []
    for d in districts:
        d = norm(d)
        if d in unique:
            hit.append(d)
        else:
            sub = [v for v in unique if d in v or v in d]
            if sub:
                hit.extend(sub)
            else:
                missing.append(d)
    return hit, missing, unique


def backup_files(paths, tag):
    """把文件（连同同名 .shx/.dbf/.prj/.cpg）移到 temp/ 备份目录。"""
    bak = os.path.join(_ROOT, "temp", "crop_backup_%s%s" % (date.today().strftime("%Y%m%d"), tag))
    os.makedirs(bak, exist_ok=True)
    moved = 0
    for p in paths:
        if not p:
            continue
        stem, _ = os.path.splitext(p)
        for f in glob.glob(stem + ".*"):
            shutil.move(f, os.path.join(bak, os.path.basename(f)))
            moved += 1
    return bak, moved


def clip_one(src, gdf_bound, out_path, label):
    if not src or not os.path.exists(src):
        print("[skip] %s 不存在: %s" % (label, src))
        return None
    gdf = gpd.read_file(src)
    n0 = len(gdf)
    if gdf.crs is not None and gdf_bound.crs is not None and gdf.crs != gdf_bound.crs:
        gdf = gdf.to_crs(gdf_bound.crs)
    gdf_c = gpd.clip(gdf, gdf_bound, keep_geom_type=True)
    n1 = len(gdf_c)
    gdf_c.to_file(out_path, encoding="utf-8")
    pct = (100.0 * n1 / n0) if n0 else 0.0
    print("      %s: %d -> %d (%.1f%%)  -> %s" % (label, n0, n1, pct, out_path))
    return (n0, n1)


def main():
    ap = argparse.ArgumentParser(
        description="按行政区边界裁剪建筑 / 地块 shp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=os.path.join(_ROOT, "config.yaml"),
                    help="配置文件路径（默认项目根 config.yaml）")
    ap.add_argument("--boundary", default=None, help="行政区界线 shp [必填]")
    ap.add_argument("--districts", nargs="+", default=None, help="要保留的区名，可给多个 [必填]")
    ap.add_argument("--district-field", default=None, help="区界 shp 里的区名字段名（默认自动探测）")
    ap.add_argument("--list", action="store_true", help="只列出区界 shp 里的区名后退出")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认与输入 shp 同目录）")
    ap.add_argument("--suffix", default="_crop", help="输出文件名后缀（默认 _crop）")
    ap.add_argument("--in-place", action="store_true", help="覆盖写回 config 指向的原文件")
    ap.add_argument("--no-backup", action="store_true", help="--in-place 时不备份（不建议）")
    args = ap.parse_args()

    if not args.boundary:
        raise SystemExit("ERROR: 缺少 --boundary（行政区界线 shp）。用 --help 看用法。")
    if not os.path.exists(args.boundary):
        raise SystemExit("ERROR: 找不到区界 shp: %s" % args.boundary)

    print("[1/4] 读取区界: %s" % args.boundary)
    gdf_bound = gpd.read_file(args.boundary)
    field = pick_district_field(gdf_bound, args.districts, args.district_field)

    if args.list:
        if field is None:
            print("      未探测到区名字段。字段列表: %s" % ", ".join(map(str, gdf_bound.columns)))
            return
        names = sorted({norm(v) for v in gdf_bound[field].astype(str)})
        print("      区名字段: %s（共 %d 个）" % (field, len(names)))
        for n in names:
            print("        - %s" % n)
        return

    if not args.districts:
        raise SystemExit("ERROR: 缺少 --districts（要保留的区名）。先用 --list 看有哪些区。")
    if field is None:
        raise SystemExit(
            "ERROR: 没能自动识别区名字段，请用 --district-field 指定。可用字段: %s"
            % ", ".join(map(str, gdf_bound.columns))
        )

    cfg = load_config(args.config)
    data_cfg = (cfg.get("data") or {})
    buildings_src = resolve_path(data_cfg.get("buildings_shp"))
    landuse_src = resolve_path(data_cfg.get("landuse_shp"))
    target_crs = data_cfg.get("crs") or "EPSG:4326"

    print("[2/4] 筛选行政区（字段: %s）..." % field)
    hit, missing, unique = match_districts(gdf_bound[field], args.districts)
    if not hit:
        raise SystemExit(
            "ERROR: 没匹配到任何区。你给的: %s\n      区界里可用: %s"
            % (args.districts, ", ".join(unique))
        )
    if missing:
        print("      [warn] 未匹配到: %s（已跳过）" % ", ".join(missing))
    sel = gdf_bound[gdf_bound[field].astype(str).map(norm).isin(hit)].copy()
    sel = sel.to_crs(target_crs)
    area_km2 = sel.area.sum() / 1e6
    print("      选中区: %s | 合计面积约 %.1f km2 | 目标坐标系: %s"
          % (", ".join(hit), area_km2, target_crs))

    tag = "_" + "_".join(hit)[:40]
    out_dir = args.out_dir or None
    plan = []
    for src, label in ((buildings_src, "建筑"), (landuse_src, "地块")):
        if not src:
            continue
        if args.in_place:
            out = src
        else:
            d = out_dir or os.path.dirname(src)
            stem = os.path.splitext(os.path.basename(src))[0]
            out = os.path.join(d, stem + args.suffix + ".shp")
        plan.append((src, label, out))

    if not plan:
        raise SystemExit("ERROR: config 里没有可用的 data.buildings_shp / data.landuse_shp。")

    if args.in_place and not args.no_backup:
        print("[3/4] 备份原文件到 temp/ ...")
        bak, moved = backup_files([p[0] for p in plan], tag)
        print("      已备份 %d 个文件 -> %s" % (moved, bak))
    else:
        print("[3/4] 跳过备份（未使用 --in-place）")

    print("[4/4] 空间裁剪并写出...")
    for src, label, out in plan:
        d = os.path.dirname(out)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        clip_one(src, sel, out, label)

    print("\n==== 完成 ====")
    if not args.in_place:
        print("  注意：结果写到了新文件，原文件未动。")
        print("  若要使用裁剪结果，请把 config.yaml 的 data.buildings_shp / data.landuse_shp")
        print("  指向上面的输出文件，再把 pipeline.run_build 设为 true 重新建图。")


if __name__ == "__main__":
    main()
