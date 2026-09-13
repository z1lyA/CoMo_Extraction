# 城市形态–功能关联 GNN 框架

> 一套「建筑形态 → 城市功能」关联分析的可运行代码框架。
> 核心思路：把**建筑底面**建成**图**（每栋建筑一个节点，相邻建筑连边，每节点带 22 维形态指标），用 GNN 学「形态 → 地块功能」，再**反解释**出关键形态特征、核心形态子类与地块构型。

---

## 1. 文件结构与职责

代码按**职责**分装到 4 个子目录：

```
CoMo_Extraction/
├── src/                        # 核心库（被多入口复用）
│   ├── Build_Graphs.py         # ①② 清洗 + 建图
│   ├── GNN_Train.py            # ③ GNN 训练（10 折 CV）
│   ├── Explain_FeatureImportance.py  # ④ XGNN 解释（F_S/G_S）
│   ├── Cluster_CoreForms.py    # ⑤ 核心形态子类聚类（T1..TK）
│   ├── Extract_PlotConfig.py   # ⑥ 地块构型回贴（主导/多样）
│   ├── visualize_core_form_samples.py  # ⑤ 可视化辅助
│   └── numpy_pickle_shim.py    # numpy 1.x ↔ 2.x 兼容垫片
│
├── pipelines/                  # 两条流水线入口
│   ├── run_pipeline.py         # A 本城主调度（串起 ①→⑥）
│   └── cross_city_transfer.py  # B 跨城市泛化（零训练）
│
├── data_prep/                  # 实验前数据准备
│   ├── prepare_city.py         # 裁 EULUC 功能标签 + 清洗 SinoBF 建筑 + 质心挂接（本地处理，支持任意城市）
│   └── crop_to_districts.py    # 按行政区裁剪建筑/地块 shp
│
└── tools/                      # 工具/调参（不在主流程）
    └── tune_umap.py            # UMAP 超参自动扫描（n_neighbors × min_dist × K）
```

各文件角色与产物：

| 文件 | 角色 | 输入 | 输出 |
|---|---|---|---|
| `pipelines/run_pipeline.py` | **统一编排器**：一条命令串起 ①→⑥ | `config.yaml` | 依次调用 `src/` 下 5 个脚本 |
| `src/Build_Graphs.py` | ①② 清洗 + 建图 | `data/*.shp` | `results/run_<ts>/01_Graphs/graphs_<ts>/`（图数据集 + `metadata.pkl`） |
| `src/GNN_Train.py` | ③ GNN 训练（10 折 CV） | 图数据集 | `results/run_<ts>/02_Experiments/GNN_gcn_<weight>_<ts>/`（best_model.pth + 指标 + 混淆矩阵） |
| `src/Explain_FeatureImportance.py` | ④ XGNN 解释 | 图数据集 + 训练好的模型 | `results/run_<ts>/03_Explainer/ep_ipt<ts>/`（F_S 关键特征集 + G_S 核心子图 + `fs_global.json` + 热力图） |
| `src/Cluster_CoreForms.py` | ⑤ 核心形态子类聚类 | 建筑形态特征（读 ④ 的 F_S + G_S） | `results/run_<ts>/04_Clustering/run_<ts>_K<k>/`（T1..TK 类型 + 自动中英命名 + 画像图） |
| `src/Extract_PlotConfig.py` | ⑥ 地块构型回贴 | ⑤ 的类型 + 全量建筑特征 | `results/run_<ts>/05_PlotConfig/run_<ts>/`（主导型/多样型 + 分布图） |
| `src/numpy_pickle_shim.py` | 兼容垫片：`numpy._core`→`numpy.core` | — | 让 numpy 1.x 能读 numpy 2.x 存的 pkl |
| `data_prep/crop_to_districts.py` | **预处理（可选）**：按行政区**边界**（非矩形）裁剪建筑/地块 shp。读 `config.yaml` 的 `data.*` 路径，区界与区名由命令行给：`--boundary <区界线.shp> --districts 区A 区B`（`--list` 可先列出可用区名，`--district-field` 兜底指定区名字段） | 行政区界线 shp（自备）+ `config.yaml` | 默认写 `data/<原名>_crop.shp`，**不动原文件**；加 `--in-place` 才覆盖原文件并自动备份到 `temp/crop_backup_<日期>/` |

### 运行产物归档（一次运行 = 一个文件夹）
通过 `run_pipeline.py` 跑时，所有步骤产物统一归档到 `results/run_<时间戳>/` 下的子目录，方便整次运行"一处查看、一处删除"：
```
results/run_20260801_075435/
├── pipeline_run_<ts>.log      # 本次运行日志
├── 01_Graphs/                 # ①② 图数据集
├── 02_Experiments/            # ③ 训练 + 模型 + 指标
├── 03_Explainer/             # ④ F_S / G_S / 热力图
├── 04_Clustering/            # ⑤ 核心形态类型 T1..TK
└── 05_PlotConfig/            # ⑥ 地块构型
```
> 各步骤脚本**也支持独立运行**（不经由 `run_pipeline.py`）：此时没有统一运行目录，会回退到旧的 `results/Graphs`、`results/Experiments` 等分类目录，行为不变。
> 第⑥步在独立运行或分步运行时，会自动在「当前运行目录」与「历史 `results/` 分类目录」中回退查找最新的 ⑤ 产物，链路不中断。

---

## 2. 全流程（6 步，一句话理解）

> 把"建筑长什么样"变成"图"，让模型学"长这样 → 是啥功能"，再反过来问"模型到底看哪些形态特征做判断"，最后把建筑聚成几类核心形态、给每个地块定性。

1. **①② 清洗 + 建图**（`Build_Graphs.py`）
   - 清洗：合并重叠建筑、剔除非建筑（细长比过大/面积过小）。
   - 建图：Delaunay 三角剖分 + 距离阈值（默认 200m）截断边；每节点 22 维形态指标；保留建筑质心供训练时实时算边权。
2. **③ 训练**（`GNN_Train.py`）
   - 3 层图卷积 + 3 层池化 + MLP，10 折分层交叉验证（或 `quick_fold` 单折）。
   - 输出最佳模型 `best_model.pth` 与逐折指标。
3. **④ 解释**（`Explain_FeatureImportance.py`）
   - 用 GNNExplainer 对**预测正确**的地块，提取：
     - **F_S**：每个地块的「关键形态指标」（≥阈值 0.95）；
     - **G_S**：每个地块的「核心子图关键建筑」；
     - 全局 `fs_global.json`：跨所有地块汇总出的**核心城市形态特征集**（步骤⑤聚类用的特征子空间）。
4. **⑤ 聚类出子类**（`Cluster_CoreForms.py`）
   - 对建筑形态特征做 UMAP 降维 + KMeans（按轮廓系数自动选最优 K，非固定 8）。
   - 每类 T1..TK 按"特征 z-score 均值 Top-N"**自动中英命名**。
5. **⑥ 地块构型**（`Extract_PlotConfig.py`）
   - 把 ⑤ 的降维器/簇心套到**全部建筑**上分类，逐地块统计构成比例：主导型（T_x-dom）/ 多样型（diverse）。
6. **编排**（`run_pipeline.py`）：按 `config.yaml` 的 `pipeline` 段开关，依次跑以上 5 步，任一步失败即停。

---

## 3. 运行方式与关键参数

全流程由**单一配置文件 `config.yaml`** 驱动，所有脚本只读它，改参数不用动代码。在项目根目录下，用你配好的 Python 环境执行编排入口即可：

```bash
python pipelines/run_pipeline.py
```

> 环境准备见 `ENVIRONMENT.md`：需自带 `torch` / PyG / `geopandas` 的环境，勿用系统默认 Python。

### 首次运行只需关心两个开关（其余保持默认即可跑通）
| 参数 | 默认 | 说明 |
|---|---|---|
| `training.quick_fold` | `false` | **`true`**=单折快速验证（几分钟跑通整条链，先确认代码可行）；**`false`**=正式 10 折交叉验证 |
| `pipeline.run_clean` | `false` | **`true`**=从原始建筑 shp 重新清洗（可复现）；**`false`**=直接读已清洗好的建筑 shp（更快） |

### 六个步骤开关（`pipeline` 段，`true`=执行、`false`=跳过，任一步失败即停）
`run_clean`(①清洗) → `run_build`(②建图) → `run_train`(③训练) → `run_explain`(④解释) → `run_cluster`(⑤聚类) → `run_plot_config`(⑥地块构型)。
> 只改了后段内容时，可只开对应开关（例：图与模型都已就绪、只想重出聚类图，就关 `run_build`/`run_train`、留 `run_explain`/`run_cluster`）。

### 换城市要改什么
改 `data.buildings_shp` / `data.landuse_shp` 指向该城市的建筑与地块 shp，并把 `data.crs` 设为对应投影坐标系即可，**无需新建 config 变体**。

> 完整参数含义与默认值见 §4「`config.yaml` 参数全解」。

---

## 4. `config.yaml` 参数全解

> 所有脚本只读这一个文件。改这里即可调参，不用动代码。

### ① `data` —— 数据路径与字段
| 参数 | 含义 | 何时改 |
|---|---|---|
| `buildings_shp` | 清洗后建筑底面 shp | 换数据源 |
| `raw_buildings_shp` | 原始建筑 shp（清洗原料） | `cleaning.enable=true` 时生效 |
| `landuse_shp` | 地块功能 shp（5 类标签） | 换数据源 |
| `*_id_field` / `*_height_field` / `*_block_id_field` | 各 shp 的 ID / 高度 / 地块ID 字段名 | 字段名变了就改 |
| `landuse_label_field` | 功能标签字段（默认 `Level1`） | 标签字段名变了 |
| `crs` | 统一投影坐标系（默认 `EPSG:32650`） | 换城市/坐标系 |

### ② `cleaning` —— 清洗（仅 `pipeline.run_clean=true` 时执行）
| 参数 | 含义 |
|---|---|
| `enable` | 是否从 raw 重新清洗（关掉则直接读已清洗 shp） |
| `enable_merge` | 合并重叠/接触建筑（取最大高度） |
| `min_area` | 面积下限(m²)，过小剔除 |
| `max_elongation` | 细长比上限，过大视为线状地物剔除 |
| `elongation_area_limit` | 细长比判定只对"面积<此值"的建筑生效 |
| `merge_buffer_distance` | 合并 buffer(m) |

### ③ `device` —— 运行设备
| 参数 | 含义 |
|---|---|
| `use_gpu` | 优先用 GPU（无 CUDA 自动回退 CPU） |
| `force_cpu` | `true` 强制 CPU（调试用） |

### ④ `graph` —— 图构建（Delaunay + 距离阈值）
| 参数 | 含义 | 何时改 |
|---|---|---|
| `distance_threshold` | 固定模式下的边距离阈值(m)，默认 **200** | `threshold_mode=fixed` 时生效 |
| `threshold_mode` | `fixed`=用上面阈值；`auto`=按边长分布自动选 | 想"看分布取阈值"用 `auto` |
| `auto_quantile` | `auto` 时取边长分布的该分位数（默认 0.95，对应"尾部密度近零"） | 调 `auto` 的截断点 |
| `centroid_merge_threshold` | 质心合并阈值 | 一般不动 |
| `min_buildings_per_block` | 单地块最少建筑数，不足跳过（避免退化图） | 数据很稀时调小 |

### ⑤ `model` —— GNN 结构
| 参数 | 含义 | 何时改 |
|---|---|---|
| `hidden_dim` | 隐藏层维度（默认 128） | 调容量 |
| `output_dim` | 输出类别数（默认 5 类功能） | **一般不动** |
| `pool_ratio` | 池化比例（SAGPooling 保留比例） | 调池化强度 |
| `dropout` | Dropout 比例 | 防过拟合 |
| `edge_weight` | `none`=不加权；`inverse`=质心反距离加权 | **想用地理边权改 `inverse`** |

### ⑥ `output` —— 图数据输出
| 参数 | 含义 |
|---|---|
| `base_output_dir` | 图数据输出根目录 |

### ⑦ `training` —— 训练超参
| 参数 | 含义 | 何时改 |
|---|---|---|
| `n_splits` | 交叉验证折数（默认 10） | 想快用 5 |
| `quick_fold` | **`true`=单折快速验证（几分钟跑通整链）** | **调试必开** |
| `epochs` | 每折最大 epoch（配合早停） | — |
| `patience` | 早停耐心（验证指标多少轮不涨则停） | 数据难学就调大 |
| `batch_size` | 批大小 | 显存/内存不够调小 |
| `learning_rate` | 学习率 | 不收敛就调 |
| `weight_decay` | L2 正则 | 过拟合调大 |
| `label_smoothing` | 标签平滑（默认 0.1，缓解过自信） | — |
| `seed` | 随机种子（保证可复现） | — |

### ⑧ `training_paths` —— 路径（建图脚本会自动写回正确的 `graph_data_dir`）
| 参数 | 含义 |
|---|---|
| `experiment_dir` | 训练产物目录 |
| `graph_data_dir` | **图数据目录**（Build_Graphs 跑完会自动更新此值，勿手改） |

### ⑨ `explainer` —— XGNN 解释（步骤④）
| 参数 | 含义 | 何时改 |
|---|---|---|
| `top_k` | 每类取的关键形态特征数；各类 Top-K 并集去重后即全局 F_S（核心城市形态特征集） | 想 3 或 5 |
| `max_samples_per_class` | 每类最多解释样本数（默认 100） | 太慢就调小 |
| `explain_epochs` | GNNExplainer 优化轮数 | — |
| `fs_threshold` | 单地块 F_S 的特征重要性阈值（默认 0.950） | 放宽/收紧单地块关键特征 |
| `gs_threshold` | G_S 节点硬阈值（一般不触发） | — |
| `gs_top_frac` | G_S 回退：节点没达阈值时取前 25% 最关键建筑 | — |

### ⑩ `clustering` —— 出核心形态子类（步骤⑤）
| 参数 | 含义 | 何时改 |
|---|---|---|
| `feature_source` | 聚类特征子空间来源：`fs_from_explainer`=读④ F_S（**默认，对齐参照研究**）；`custom`=下面 custom_features；`all`=全 22 维 | 主要开关 |
| `custom_features` | `custom` 模式下的特征子集（参照研究子类使用的形态维度） | 指定特征 |
| `umap_n_neighbors` | UMAP 邻居数（建议 15–50） | 调降维 |
| `umap_min_dist` | UMAP 最小距离（越小簇越散） | 调降维 |
| `min_k` / `max_k` | KMeans 聚类数 K 的下/上界（按轮廓系数自动选最优 K，**非固定 8**） | 调 K 搜索范围 |
| `use_gs_buildings` | `true`=聚类只用 ④ 提取的核心建筑（G_S）而非全部建筑 | — |
| `random_state` | 随机种子 | — |
| `num_representative` | 每类代表建筑数（命名/出图用） | — |
| `name_top_n` | 自动命名取"|z-score 均值| 最大"的前 N 个特征拼接成描述名 | 调命名粒度 |
| `extract_shp` | 是否导出代表性建筑 shp | — |
| `top_n_per_type` | 每类取 top-N 最显著特征构建核心特征集 | 调特征集大小 |

### ⑪ `plot_config` —— 地块构型（步骤⑥）
| 参数 | 含义 |
|---|---|
| `dominant_threshold_general` | 一般地块主导阈值（默认 0.50） |
| `dominant_threshold_residential` | 住宅地块主导阈值（更严，0.60，对齐参照研究） |
| `min_dominant_buildings` | 主导型最少关键建筑数（少于判为 diverse） |
| `residential_class_names` | 命中这些功能名（按数据的标签取值，住宅=`居住`/`1.0`）则套用更严阈值 |
| `extract_shp` | 是否按 block_id 关联 function.shp 导出空间落图 |

### ⑫ `pipeline` —— 全流程总开关（run_pipeline.py 用）
| 参数 | 含义 |
|---|---|
| `run_clean` | 步骤① 清洗 |
| `run_build` | 步骤② 建图 |
| `run_train` | 步骤③ 训练 |
| `run_explain` | 步骤④ 解释 |
| `run_cluster` | 步骤⑤ 聚类 |
| `run_plot_config` | 步骤⑥ 构型 |

---

## 5. 跨城市：数据准备与泛化推理（prepare_city.py + cross_city_transfer.py）

本框架已支持「换城市」与「跨城市零样本泛化」两个扩展方向，核心是把早期覆盖有限的建筑轮廓替换为权威的 **SinoBF-1** 建筑级几何，功能标签仍用权威的 **EULUC-China 2.0**。

### 5.1 prepare_city.py —— 实验前准备（本地模式，不联网下载）

本脚本**只做本地处理，不联网、不自动下载**。你先自己准备好两类数据（见 §5.1.1），再用本脚本做「裁 EULUC 功能标签 + 清洗 SinoBF 建筑 + 质心挂接」三步：

- **Step 1** 读本地 SinoBF-1 建筑轮廓（.shp 或 .zip）→ 清洗（剔空几何 + buffer(0) 修无效几何）。
- **Step 2** 读本地全国 EULUC-China 2.0（.shp 或 .zip）→ 按城市 bbox 裁出 `data/function_<city>.shp`，并把 EULUC 的 `Class(0-10)` 映射成 5 大类 `Level1` + 生成 `block_id`。
- **Step 3** 建筑质心 within 连 EULUC 地块挂 `block_id` → 写出 `data/sinobf_<city>_buildings.shp`，流水线可直接读取。

设计原则：**幂等**（本地已有则跳过，`--force` 强制重解压）+ **可控**（用 `--sinobf-zip/--euluc-zip/--func-shp` 等指定本机文件）。任一必需文件缺失会清晰报错并提示去 Zenodo 下，绝不联网。

#### 5.1.1 你需要准备的两类数据（建议来源）
1. **城市形态（建筑轮廓）**：SinoBF-1 某城市 zip，Zenodo 17844789，命名 `G_P_C.zip`（大区 + 省 + 市）。用浏览器下（脚本不联网）。
2. **地块功能标签**：全国 EULUC-China 2.0 zip（约 3.14 GB，Zenodo 16794007，文件 `EULUC_China_20_SHP.zip`，解压后 `_1`/`_2` 两片）。本机只需一份，任意城市复用。

```powershell
# 标准：建筑 zip + 全国 EULUC zip（均本机已有），离线跑
python data_prep/prepare_city.py --city Shenzhen `
    --sinobf-zip "data/South_Guangdong_Shenzhen_SinoBF-1.zip" `
    --euluc-zip "data/EULUC_China_20_SHP.zip"

# 功能标签已自己裁好：跳过 Step 2
python data_prep/prepare_city.py --city Shenzhen --sinobf-zip "data/South_Guangdong_Shenzhen_SinoBF-1.zip" --func-shp "data/function_Shenzhen.shp"
```
> 完成后，把 `config.yaml` 的 `data.buildings_shp` / `landuse_shp` 指向产出文件，即可接入主流程。

### 5.2 pipelines/cross_city_transfer.py —— 跨城市泛化推理（流水线 B）
用**你自己训练好的源城市 `best_model.pth`** 直接推断**目标城市**图，验证泛化性。两个对齐难点已处理：
1. **特征空间对齐**：各城市建图时用自己的 `StandardScaler`，推断前先 `tgt_scaler.inverse_transform` 还原原始特征，再 `src_scaler.transform` 投到源城市缩放空间（scaler 存在各自 `metadata.pkl`）。
2. **标签空间对齐**：用源城市 `label_encoder` 把真实标签与预测索引统一到源城市索引空间；目标城市出现源城市没有的类别会标 `-1` 并跳过。

最简用法（数据齐后）：
```powershell
# (A) 自测：源 = 目标（同一城市），应复现该城精度，验证脚本逻辑
python pipelines/cross_city_transfer.py `
  --src_config config.yaml `
  --src_graph_dir results/<源城图目录>/graphs_xxx `
  --src_model results/<源城实验目录>/GNN_gcn_none_xxx/best_model.pth `
  --tgt_graph_dir results/<源城图目录>/graphs_xxx `
  --out results/transfer_self_test

# (B) 真实跨城市（源城模型 -> 目标城）：先 Build_Graphs 构建目标城图，再跑
python src/Build_Graphs.py config.yaml   # 记下 graphs_xxx 目录（或让 cross_city_transfer 用 --tgt_config 自动建图）
python pipelines/cross_city_transfer.py `
  --src_config config.yaml --src_graph_dir <源城图目录> --src_model <源城 best_model.pth> `
  --tgt_graph_dir <目标城图目录> --out results/transfer_src2tgt
```
产物：`transfer_metrics.json / .csv`（Acc / macro-F1 / weighted-F1）+ `transfer_predictions.shp`。
> 示例：以广州为源、深圳为目标的一次实测 Acc 0.6369 / macro-F1 0.3476 / weighted-F1 0.5972（全居住基线 57.7%）。少数类（商业/交通）recall≈0 塌缩，是后续可优化点。

---

## 6. 已知约束 / 踩坑

- **numpy 版本**：运行环境必须 `numpy==1.24.4`；升 2.x 会让 torch 2.2.0 直接崩（`Numpy is not available`）。`metadata.pkl` 是 numpy 2.x 存的，靠 `numpy_pickle_shim.py` 兼容读取。
- **GNNExplainer 固定走 node_mask**：SAGPooling 与 edge_mask 不兼容（PyG 2.8 下会崩），故解释器固定用 node_mask 定义节点重要性，无需在 config 里配置。
- **类别不平衡**：部分功能类别（如工业、交通）样本极少，这两类基本学不动（macro-F1 被拉低）。这是精度瓶颈，不是池化问题。
