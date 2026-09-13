#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py —— 统一流水线入口：一条命令串起全流程
    清洗 → 建图 → 训练 → 解释(④产F_S+G_S) → 出子类(⑤) → 回贴G_S定地块构型(⑥)

用法：
    python run_pipeline.py                 # 用默认 config.yaml，按 pipeline 段开关跑
    python run_pipeline.py config.yaml     # 指定配置文件

设计：
    - 各子步骤复用现有脚本（Build_Graphs / GNN_Train / Explain_FeatureImportance /
      Cluster_CoreForms / Extract_PlotConfig），全部读同一个 config.yaml，保证「一个 yaml 调参」。
    - pipeline 段（config.yaml）控制每步是否执行；任一步失败即停，并打印日志位置。
    - 清洗开关(pipeline.run_clean)会同步到 cleaning.enable，再交给 Build_Graphs 执行。

快速验证：把 config.yaml 的 training.quick_fold 设为 true，即可单折分钟级跑通整条链。
"""

import os
import re
import sys

import sys as _sys
try:
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(errors='replace')
        _sys.stderr.reconfigure(errors='replace')
except Exception:
    pass
import time
import yaml
import subprocess
import datetime
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                # 项目根目录（src/ / pipelines/ / data_prep/ / tools/ 的父目录）
PY = sys.executable  # 用当前（managed venv）python 跑所有子步骤，保证环境一致


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_cleaning_enable(config_path, value):
    """逐行替换 cleaning.enable 的布尔值，保留行内注释与全部其他注释。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 只替换 `enable: <bool>` 中的布尔值，保留后面的行内注释
    new_text, n = re.subn(
        r'^(\s*enable:\s*)(true|false)',
        lambda m: f"{m.group(1)}{str(value).lower()}",
        text, count=1, flags=re.MULTILINE
    )
    if n:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_text)
        return True
    return False


def run_step(step_name, cmd, log_f):
    """运行一个子步骤；失败返回非零。

    关键：子进程【继承】本进程的 stdout/stderr（即当前用户终端），不重定向到管道。
    这样 GNN_Train / Cluster 等脚本里的 tqdm 能识别到真实 tty，用 \\r 原地刷新进度条，
    既不会刷屏重复打印、也不会被静默关掉。
    （Windows 无标准 pty；若用 PIPE 包住子进程，tqdm 会退化成逐行打印或自动禁用，
     这正是之前"重复刷行 / 进度条消失"两种现象的根因。）
    子步骤的运行细节由各脚本自己落盘（best_model.pth / metrics / png 等），本处仅记录步骤起止与退出码。
    """
    print(f"\n{'='*70}\n>>> 步骤 {step_name}\n{'='*70}")
    log_f.write(f"\n### STEP {step_name} ###\n启动命令: {' '.join(cmd)}\n")
    log_f.flush()
    t0 = time.time()
    # stdout=None -> 继承用户终端；tqdm 进度条在真实 tty 下原地刷新
    proc = subprocess.Popen(cmd, cwd=ROOT)
    proc.wait()
    dt = time.time() - t0
    log_f.write(f"步骤 {step_name} 结束：退出码 {proc.returncode}，耗时 {dt:.1f}s\n")
    log_f.flush()
    if proc.returncode != 0:
        print(f"[失败] 步骤 {step_name} 失败（退出码 {proc.returncode}，耗时 {dt:.1f}s）")
        return False
    print(f"[完成] 步骤 {step_name} 完成（耗时 {dt:.1f}s）")
    return True


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config_path = os.path.join(ROOT, config_path) if not os.path.isabs(config_path) else config_path
    if not os.path.exists(config_path):
        print(f"错误: 未找到配置文件 {config_path}")
        return

    cfg = load_config(config_path)
    pl = cfg.get('pipeline', {})

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_dir = os.path.join(results_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.environ['PIPELINE_RUN_DIR'] = run_dir   # 传给各子脚本，使其归档到同一运行目录
    log_path = os.path.join(run_dir, f"pipeline_run_{ts}.log")
    log_f = open(log_path, 'w', encoding='utf-8')

    print(f"=== 统一流水线启动 [{ts}] ===")
    print(f"配置文件: {config_path}")
    print(f"运行日志: {log_path}")
    print(f"运行归档目录: {run_dir}")
    print(f"步骤开关: clean={pl.get('run_clean')} build={pl.get('run_build')} "
          f"train={pl.get('run_train')} explain={pl.get('run_explain')} cluster={pl.get('run_cluster')} "
          f"plot_config={pl.get('run_plot_config')}")
    print(f"quick_fold={cfg['training'].get('quick_fold')}  threshold_mode={cfg['graph'].get('threshold_mode')}")

    # ---- 步骤①+② 清洗 + 建图（Build_Graphs 同时做两件事）----
    if pl.get('run_build', True):
        # 同步清洗开关：pipeline.run_clean -> cleaning.enable（逐行替换，保留注释）
        set_cleaning_enable(config_path, bool(pl.get('run_clean', True)))
        ok = run_step("①② 清洗+建图", [PY, '-m', 'src.Build_Graphs', config_path], log_f)
        if not ok:
            log_f.close(); return
        # Build_Graphs 已把最新 graph_data_dir 写回 config.yaml，重新读取供后续步骤使用
        cfg = load_config(config_path)
    else:
        print("\n>>> 跳过 ①②（run_build=false）：复用已有图数据 "
              f"({cfg['training_paths'].get('graph_data_dir')})")

    # ---- 步骤③ 训练 ----
    if pl.get('run_train', True):
        ok = run_step("③ 训练", [PY, '-m', 'src.GNN_Train', config_path], log_f)
        if not ok:
            log_f.close(); return
    else:
        print("\n>>> 跳过 ③（run_train=false）")

    # ---- 步骤④ 解释 ----
    if pl.get('run_explain', True):
        ok = run_step("④ 解释(XGNN)", [PY, '-m', 'src.Explain_FeatureImportance', config_path], log_f)
        if not ok:
            log_f.close(); return
    else:
        print("\n>>> 跳过 ④（run_explain=false）")

    # ---- 步骤⑤ 出子类 ----
    if pl.get('run_cluster', True):
        ok = run_step("⑤ 出子类(聚类)", [PY, '-m', 'src.Cluster_CoreForms', config_path], log_f)
        if not ok:
            log_f.close(); return
    else:
        print("\n>>> 跳过 ⑤（run_cluster=false）")

    # ---- 步骤⑥ 回贴 G_S 定地块构型（主导/多样）----
    if pl.get('run_plot_config', True):
        ok = run_step("⑥ 定地块构型(G_S回贴)", [PY, '-m', 'src.Extract_PlotConfig', config_path], log_f)
        if not ok:
            log_f.close(); return
    else:
        print("\n>>> 跳过 ⑥（run_plot_config=false）")

    log_f.close()
    print(f"\n=== 流水线全部完成！日志: {log_path} ===")


if __name__ == "__main__":
    main()
