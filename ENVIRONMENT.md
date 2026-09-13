# 环境配置 / Environment Setup

> 本框架在 **Windows + CPU（无 CUDA）** 下开发与验证。下面给出可复现的依赖组合与若干 Windows 专属铁律。

## 1. Conda 环境（推荐）

实测可用组合：Python 3.11 + CPU。先建环境，再装依赖：

```bash
# 1) 建环境（环境名可自定义，下文以 comognn 为例）
conda create -n comognn python=3.11 -y
conda activate comognn

# 2) 装核心（含 torch 2.2.0 CPU 版）
pip install -r requirements.txt

# 3) 装 PyG 的底层扩展（不在 PyPI 普通索引，需指定官方 wheel）
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

## 2. 依赖锁版本（必读）

`requirements.txt` 已锁定 proven-good 版本。其中两条是**铁律**：

- **`numpy==1.24.4` 必须锁死**。升到 `numpy>=2.x` 会让 `torch 2.2.0` 的 C 扩展初始化失败（报 `Numpy is not available` / `Failed to initialize NumPy`）。
- 如需读取由 numpy 2.x 存出的 `*.pkl`，仓库内 `numpy_pickle_shim.py` 垫片会自动把 `numpy._core` 重定向到 `numpy.core`，无需降回 numpy 2.x。

> 环境里还要有 `geopandas`（读写 shp）、`umap-learn`（降维）、`scikit-learn`（KMeans/轮廓系数）、`numba`/`pynndescent`（UMAP 后端）。若 UMAP 在 Windows 上原生崩溃，优先升级 `numba`，**勿动 numpy/torch**。

## 3. Windows 运行铁律（踩过两次坑，发布前必守）

1. **禁止任何非 ASCII 字符进 `print` / stdout**。Windows 子进程 stdout 是 GBK 管道，emoji 与生僻汉字会触发 `UnicodeEncodeError` 直接崩、丢掉已训练产物。所有主脚本已清除 emoji，并在 `import sys` 后注入 `sys.stdout/stderr.reconfigure(errors='replace')` 兜底（漏网字符变 `?` 不崩）。**新增 print 一律只用 ASCII。**
2. **关键产物保存必须在展示性打印/绘图之前**：先 `torch.save` 模型 + 写指标 csv/json，再画图/打印总结。曾因保存排在绘图 print 之后，4.3h 训练因绘图输出崩溃而权重全丢。
3. **tqdm 进度条**：tqdm 能否原地刷新只取决于「子进程 stdout 是否为真实 tty」。`pipelines/run_pipeline.py` 用 `subprocess.Popen(cmd, cwd=HERE)` 继承用户终端 stdout（不重定向），子进程连真实 tty，tqdm 自然 `\r` 原地刷新。Windows 无标准 pty，伪造 `isatty` 包装无效。

## 4. 数据准备的网络注意点

- **`prepare_city.py` 不联网下载**。由于 Zenodo 直连可能受限，脚本采用纯本地处理：先用浏览器从 Zenodo 下好数据，再用 `--sinobf-zip` / `--euluc-zip` / `--func-shp` 把本机文件喂给脚本（见 README §5.1）。
- 若某文件缺失，脚本会清晰报错并提示去 Zenodo 取，不会静默联网。

## 5. 跑通一条链的最快方式

在项目根目录下，用你装好依赖的环境执行：

```bash
conda activate comognn          # 换成你的环境名

# 单折快速验证（几分钟跑通整条链，不需要 GPU）：
# 把 config.yaml 的 training.quick_fold 设为 true，然后
python pipelines/run_pipeline.py

# 也可显式指定配置文件（切换城市只需改 data.* 路径，无需 config 变体）
python pipelines/run_pipeline.py config.yaml
```

> 注意：`config.yaml` 里的 `training_paths.graph_data_dir` 是占位空串，**由 `Build_Graphs`（步骤②建图）自动写回真实路径**，请勿手填。
