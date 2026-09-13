# 数据来源与引用 / Data Sources & Citations

本框架用到的空间数据分三类：**建筑几何（轮廓）**、**地块功能标签**、**方法参照论文**。下面给出出处、获取方式与引用要求。

## 1. 建筑轮廓：SinoBF-1（权威、建筑级）

- **是什么**：首个中国**建筑级**功能地图，覆盖 109 城、约 1.1 亿栋建筑，含建筑多边形 + 8 类功能（居住/商业/工业/医疗/文体/教育/公共服务/行政）。标注基于 EULUC-China + OSM。
- **为什么用**：早期使用的建筑轮廓数据覆盖范围有限、来源不够权威。SinoBF-1 提供权威建筑多边形；我们**只取其几何（轮廓），不取它的建筑类别**——GNN 的 22 维形态指标全部由几何派生（面积/周长/凸度/分形维数/最大圆度/高度/覆盖率…），功能标签仍用下面第 2 类的 EULUC。
- **获取**：
  - 论文：Li et al. 2026, *Nature Communications* 17:2827, **DOI 10.1038/s41467-026-69589-5**
  - 数据集 Zenodo：**[https://zenodo.org/records/17844789](https://zenodo.org/records/17844789)**（DOI: [10.5281/zenodo.17844789](https://doi.org/10.5281/zenodo.17844789)）
  - 代码/下载脚本：[https://github.com/LiZhuoHong/SinoBF-1](https://github.com/LiZhuoHong/SinoBF-1)
  - 命名规则 `G_P_C.zip`：大区(G) + 省(P) + 市(C)。例：广州 = `South_Guangdong_Guangzhou_SinoBF-1.zip`；武汉 = `Central_Hubei_Wuhan_SinoBF-1.zip`。湖北/海南因 Zenodo 100 文件上限被合并成 `Central_Hubei.zip` / `South_Hainan.zip`。
- **引用**：使用 SinoBF-1 必须引用上述论文与数据集。

## 2. 地块功能标签：EULUC-China 2.0（权威、地块级）

- **是什么**：全国地块级（parcel-level）土地利用数据，5 类功能标签（居住/商业/工业/交通/公共管理），已是**权威功能真值**，作为 GNN 的训练目标（target）。
- **重要澄清**：EULUC-China 2.0 是**地块级**，不含单体建筑多边形；而 SinoBF-1 是**建筑级**（更细）。因此流水线的唯一工程衔接是：把 EULUC 标签**挂回** SinoBF 多边形（建筑落在其所属 parcel 内即继承该 parcel 标签；一 parcel 含多建筑则同标签）。`Build_Graphs.py` 已有「function.shp → 建筑」空间连接逻辑，改指向新几何即可复用。
- **获取**：
  - Zenodo：**[https://zenodo.org/records/16794007](https://zenodo.org/records/16794007)**（约 3.14 GB，文件 `EULUC_China_20_SHP.zip`；分 `_1`/`_2` 两瓦片，仅含 `Class` + 几何）
  - 我们的映射（EULUC `Class` → 本项目 5 类 `Level1`，`Level1` 存为 float 以对齐跨城市 `LabelEncoder`）：
    | Class | 含义 | Level1 |
    |---|---|---|
    | 0 | 居住 | 1（住宅） |
    | 1, 2 | 商业 | 2（商业） |
    | 3 | 工业 | 3（工业） |
    | 4, 5 | 交通 | 4（交通） |
    | 6–10 | 公共管理/其他 | 5（公共管理） |
- **引用**：使用 EULUC-China 2.0 请引用其原始发布（Zenodo 16794007 对应论文/数据说明）。

## 3. 方法参照：Chen et al. (2025) 城市形态–功能 GNN 研究

- Chen, D., Feng, Y., Li, X., Qu, M., Luo, P., Meng, L. (2025). *Interpreting core forms of urban morphology linked to urban functions with explainable graph neural network*, **Computers, Environment and Urban Systems**, 118:102267.（以波士顿为例的城市形态–功能 GNN 研究）
- 本框架的方法路线与该文一致：GCN + SAGPool + Jumping Knowledge 图编码 / Delaunay 三角剖分 + 200m 距离阈值建边 / GNNExplainer 提取关键形态特征(F_S) + 梯度归因提取核心建筑(G_S) / UMAP + KMeans 聚类出核心形态子类。

## 4. 数据准备分工（本地模式，不联网）

`prepare_city.py` **不联网下载**，所有原始数据由你用浏览器从 Zenodo 下好后，用 `--sinobf-zip` / `--euluc-zip` / `--func-shp` 喂入。脚本只做本地处理：

| 步骤 | 数据 | 来源（你先下好） | 脚本产物 |
|---|---|---|---|
| 1 | 城市建筑轮廓 | SinoBF-1（Zenodo 17844789） | 解压 + 清洗 shp |
| 2 | 全国功能标签 + 按 bbox 裁剪 + `Class→Level1` 映射 | EULUC-China 2.0（Zenodo 16794007） | `data/function_<city>.shp` |
| 3 | 建筑清洗 + 质心挂 block_id | 本地计算 | `data/sinobf_<city>_buildings.shp` |

> 全国 EULUC 3.14 GB 只需本机存一份（解压后 `_1`/`_2` 两片），任意城市复用，不用重复下。详见 `README.md` §5.1。

## 5. 引用汇总（使用本框架时请在 README / 论文中引用下列来源）

```bibtex
@article{li2026sinobf,
  title   = {SinoBF-1: The first building-level functional map of China},
  author  = {Li, Zhuohong and others},
  journal = {Nature Communications},
  year    = {2026}, volume = {17}, pages = {2827},
  doi     = {10.1038/s41467-026-69589-5}
}
@dataset{euluc2020china,
  title = {EULUC-China 2.0},
  year  = {2020}, publisher = {Zenodo},
  doi   = {10.5281/zenodo.16794007}
}
@article{chen2025urbanform,
  title   = {Interpreting core forms of urban morphology linked to urban functions with explainable graph neural network},
  author  = {Chen, Dongsheng and Feng, Yu and Li, Xun and Qu, Mingya and Luo, Peng and Meng, Liqiu},
  journal = {Computers, Environment and Urban Systems},
  year    = {2025}, volume = {118}, pages = {102267}
}
```
