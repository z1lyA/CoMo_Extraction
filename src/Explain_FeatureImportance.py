#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解释器（重置版）：形态特征重要性矩阵 + 每类核心形态特征

来源：temp/2026-07-27/4explainer_importance.py（已对齐当前模型架构）
重置要点：
1. 模型类 HierarchicalGNN 与 GNN_Train.py 逐字一致（真 JumpKnowledge cat+投影、
   边权 inverse/none、forward 接收 centroids），可正确加载新 checkpoint；
2. 路径全部从 config.yaml 读取（graph_data_dir / experiment_dir），并自动发现
   最新的 best_model.pth，无需手改路径；
3. TOPK（每类核心特征数）可配置，默认 4（论文原文为"提取最重要的多个形态特征"，
   未写死 3/4；用户可在 config 的 explainer.top_k 改 3 或 5）；
4. 解释时向模型传入 centroids，使 inverse 边权在解释阶段也生效；
5. 保留原版的失败隔离与状态清理（GNNExplainer + SAGPooling 兼容性较差时的鲁棒性）。

输出（results/Explainer_Importance/<时间戳>/）：
- feature_importance_by_class.csv            各类别 × 各特征 的重要性矩阵
- feature_importance_by_class_normalized.csv 按行归一化（用于热力图）
- feature_importance_topk.csv                每类 Top-K 核心形态特征（核心城市形态指标）
- feature_importance_heatmap.png              热力图
- heatmap_data.pkl / .csv                    热力图原始数据
- feature_importance_summary.json            统计与配置快照
- explainer_failures.json                    解释失败的样本
- gnnexplainer_core.json                     ★ 每个正确预测地块的 F_S(关键形态指标,≥阈值) +
                                            G_S(核心子图关键建筑,边重要性≥阈值)；步骤⑥回贴定地块构型用
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
import pickle
import datetime
from collections import defaultdict

import numpy as np
from . import numpy_pickle_shim  # 兼容 numpy2.x 保存的 pkl（numpy._core 别名到 core）
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool, global_max_pool, JumpingKnowledge
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['font.serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_ID_TO_CN = {
    1.0: "居住",
    2.0: "商业",
    3.0: "工业",
    4.0: "交通",
    5.0: "公共管理与服务"
}

FEATURE_NAME_CN = {
    "Area": "面积",
    "Perimeter": "周长",
    "Longest_chord_length": "最长弦长",
    "Mean_radius": "平均半径",
    "Covered_area_ratio": "覆盖率",
    "Height": "高度",
    "SBRO": "短边相对方向",
    "LCO": "最长弦方向",
    "Bisector_orientation": "平分线方向",
    "Weighted_orientation": "加权方向",
    "Complexity_index": "复杂度指数",
    "IPQ": "等周商",
    "Fractality": "分形维数",
    "Max_circularity": "最大圆度",
    "Gibbs_compactness": "吉布斯紧凑度",
    "Elongation_index": "伸长指数",
    "Ellipticity": "椭圆度",
    "Solidity": "凸度",
    "DCM": "数字紧凑度",
    "Exchange_index": "交换指数",
    "BCI": "博伊斯-克拉克形状指数",
    "Cross_moment_mu11": "交叉矩",
    "Variance_spreading": "铺展方差"
}


# ============================================================
#  模型（与 GNN_Train.py 逐字一致）
# ============================================================
class HierarchicalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 pool_ratio=0.3, dropout=0.5, edge_weight_mode='inverse'):
        super(HierarchicalGNN, self).__init__()
        self.edge_weight_mode = edge_weight_mode
        self.dropout_rate = dropout

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.pool1 = SAGPooling(hidden_dim, ratio=pool_ratio)
        self.pool2 = SAGPooling(hidden_dim, ratio=pool_ratio)
        self.pool3 = SAGPooling(hidden_dim, ratio=pool_ratio)

        self.jk = JumpingKnowledge(mode='cat')
        self.jk_proj = nn.Linear(3 * hidden_dim * 2, hidden_dim * 2)

        mlp_input_dim = hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def _edge_weight(self, edge_index, centroids):
        if self.edge_weight_mode == 'none' or centroids is None:
            return None
        if edge_index.numel() == 0:
            return None
        row, col = edge_index
        dist = torch.norm(centroids[row] - centroids[col], dim=1) + 1e-6
        if self.edge_weight_mode == 'inverse':
            return 1.0 / dist
        return dist

    def _readout(self, x, batch):
        return torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=1)

    def forward(self, x, edge_index, batch, centroids=None):
        ew = self._edge_weight(edge_index, centroids)
        x = self.conv1(x, edge_index, ew)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if x.size(0) > 1:
            x, edge_index, _, batch, perm, _ = self.pool1(x, edge_index, None, batch)
            if centroids is not None and perm is not None:
                centroids = centroids[perm]
        out1 = self._readout(x, batch)

        ew = self._edge_weight(edge_index, centroids)
        x = self.conv2(x, edge_index, ew)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if x.size(0) > 1:
            x, edge_index, _, batch, perm, _ = self.pool2(x, edge_index, None, batch)
            if centroids is not None and perm is not None:
                centroids = centroids[perm]
        out2 = self._readout(x, batch)

        ew = self._edge_weight(edge_index, centroids)
        x = self.conv3(x, edge_index, ew)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if x.size(0) > 1:
            x, edge_index, _, batch, perm, _ = self.pool3(x, edge_index, None, batch)
            if centroids is not None and perm is not None:
                centroids = centroids[perm]
        out3 = self._readout(x, batch)

        out = self.jk([out1, out2, out3])
        out = self.jk_proj(out)
        return self.mlp(out)


# ============================================================
#  工具函数
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("无法从 checkpoint 中解析 state_dict。")


def infer_checkpoint_dims(state_dict):
    conv1_weight = state_dict["conv1.lin.weight"]
    input_dim = int(conv1_weight.shape[1])
    hidden_dim = int(conv1_weight.shape[0])
    output_dim = int(state_dict["mlp.8.weight"].shape[0])
    return input_dim, hidden_dim, output_dim


def get_label_names(meta, output_dim):
    if isinstance(meta, dict):
        if "label_encoder" in meta and hasattr(meta["label_encoder"], "classes_"):
            names = [str(x) for x in meta["label_encoder"].classes_]
            if len(names) == output_dim:
                return names
        if "class_names" in meta:
            names = [str(x) for x in meta["class_names"]]
            if len(names) == output_dim:
                return names
    return [str(i + 1) for i in range(output_dim)]


def get_feature_names(meta, input_dim):
    if isinstance(meta, dict) and "feature_names" in meta:
        names = [str(x) for x in meta["feature_names"]]
        if len(names) == input_dim:
            return names
    return [f"feat_{i}" for i in range(input_dim)]


def normalize_rows(df):
    arr = df.values.astype(float)
    row_max = arr.max(axis=1, keepdims=True)
    row_min = arr.min(axis=1, keepdims=True)
    denom = np.where((row_max - row_min) == 0, 1.0, (row_max - row_min))
    norm = (arr - row_min) / denom
    return pd.DataFrame(norm, index=df.index, columns=df.columns)


def make_explainer(model, explain_epochs, edge_mask_type=None):
    # 注意：edge_mask_type="object" 与含 SAGPooling 的模型不兼容（PyG 2.8 下，
    # GNNExplainer 会把"原始边数"的边掩码注入每一层卷积，但 SAGPooling 池化后
    # 边数减少，导致 message_passing 中断言失败）。因此默认关闭边掩码，
    # 改用 node_mask 的节点重要性来提取 G_S（核心子图关键建筑）。
    # 仅当模型不含 SAGPooling 时，才可设 edge_mask_type="object"。
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explain_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=edge_mask_type,
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )


def plot_heatmap_with_cn_labels(norm_df, feature_names, output_path, dpi=600):
    feature_names_cn = [FEATURE_NAME_CN.get(f, f) for f in feature_names]
    class_names_cn = []
    for idx in norm_df.index:
        try:
            class_id = float(idx)
            class_names_cn.append(CLASS_ID_TO_CN.get(class_id, idx))
        except (ValueError, TypeError):
            class_names_cn.append(CLASS_ID_TO_CN.get(idx, idx))

    norm_df_cn = norm_df.copy()
    norm_df_cn.columns = feature_names_cn
    norm_df_cn.index = class_names_cn

    fig, ax = plt.subplots(figsize=(18, max(4, 1.2 * len(norm_df_cn.index))))
    sns.heatmap(
        norm_df_cn, cmap="YlOrRd", linewidths=0.3, linecolor="white",
        cbar_kws={'label': '重要性'}, ax=ax
    )
    ax.set_xlabel("形态特征指标", fontsize=14, fontweight='bold')
    ax.set_ylabel("城市功能类别", fontsize=14, fontweight='bold')
    ax.set_title("形态特征重要性矩阵\n（基于可解释图神经网络GNNExplainer）",
                 fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"[完成] 热力图已保存: {output_path}")
    plt.close()


def save_heatmap_data(norm_df, feature_names, class_names, output_dir):
    heatmap_data = {
        'normalized_matrix': norm_df,
        'feature_names': feature_names,
        'class_names': class_names,
        'feature_mapping': FEATURE_NAME_CN,
        'class_mapping': CLASS_ID_TO_CN
    }
    heatmap_pkl_path = os.path.join(output_dir, 'heatmap_data.pkl')
    with open(heatmap_pkl_path, 'wb') as f:
        pickle.dump(heatmap_data, f)
    print(f"[完成] 热力图原始数据已保存: {heatmap_pkl_path}")
    heatmap_csv_path = os.path.join(output_dir, 'heatmap_data_normalized.csv')
    norm_df.to_csv(heatmap_csv_path, encoding='utf-8-sig')
    print(f"[完成] 热力图数据CSV已保存: {heatmap_csv_path}")


def clear_explain_state(model):
    for module in model.modules():
        if hasattr(module, "_explain"):
            module._explain = False
        if hasattr(module, "_edge_mask"):
            module._edge_mask = None
        if hasattr(module, "_loop_mask"):
            module._loop_mask = None
        if hasattr(module, "_apply_sigmoid"):
            module._apply_sigmoid = True


def compute_explainer_feature_importance(model, graph, device, explain_epochs, edge_mask_type=None):
    clear_explain_state(model)
    explainer = make_explainer(model, explain_epochs, edge_mask_type=edge_mask_type)
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    centroids = getattr(graph, 'centroids', None)
    if centroids is not None:
        centroids = centroids.to(device)
    try:
        if centroids is not None:
            explanation = explainer(x, edge_index, batch=batch, centroids=centroids)
        else:
            explanation = explainer(x, edge_index, batch=batch)
        node_mask = getattr(explanation, "node_mask", None)
        edge_mask = getattr(explanation, "edge_mask", None)
        if node_mask is None:
            raise RuntimeError("explanation.node_mask 不存在，无法做特征级聚合。")
        node_mask_np = node_mask.detach().cpu().numpy()
        edge_mask_np = edge_mask.detach().cpu().numpy() if edge_mask is not None else None
        # 特征级重要性：各建筑节点重要性按特征取均值 -> 该地块的关键形态指标得分
        if node_mask_np.ndim == 1:
            feat_scores = node_mask_np
        else:
            feat_scores = node_mask_np.mean(axis=0)
        return {
            "feat_scores": np.asarray(feat_scores, dtype=float),
            "node_mask": node_mask_np,
            "edge_mask": edge_mask_np,          # (num_edges,) 或 None
            "num_nodes": int(x.size(0)),
        }
    finally:
        clear_explain_state(model)


def find_latest_checkpoint(experiment_dir):
    cands = glob.glob(os.path.join(experiment_dir, 'GNN_*', 'best_model.pth'))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def extract_fs_gs(exp, graph, feature_names, fs_threshold, gs_threshold, top_k=4, gs_top_frac=0.25):
    """
    从 GNNExplainer 输出中提取 F_S(关键形态指标) 与 G_S(核心子图关键建筑)。
    因 SAGPooling 与 edge_mask 不兼容，G_S 改用 node_mask 的节点重要性来定（关键建筑）。
    返回 dict: {F_S(特征名列表), F_S_full(全特征重要性), G_S_node_indices, G_S_building_ids}
    """
    # ---- F_S：关键形态指标（特征级重要性）----
    feat_scores = np.asarray(exp["feat_scores"], dtype=float)
    fs_mask = feat_scores >= fs_threshold
    if fs_mask.any():
        fs_feats = [feature_names[i] for i in range(len(feature_names)) if fs_mask[i]]
    else:
        # 阈值过严导致为空 -> 回退取 Top-K 特征，保证 F_S 非空
        order = np.argsort(-feat_scores)
        fs_feats = [feature_names[i] for i in order[:top_k]]
    fs_full = {feature_names[i]: float(feat_scores[i]) for i in range(len(feature_names))}

    # ---- G_S：核心子图关键建筑（节点重要性）----
    # 节点重要性 = 各特征掩码均值；优先取 >= gs_threshold 的节点，
    # 若为空则回退取前 gs_top_frac 比例的节点，保证 G_S 非空。
    gs_nodes, gs_ids = [], []
    node_mask = exp.get("node_mask")
    if node_mask is not None:
        nm = np.asarray(node_mask, dtype=float)
        node_imp = nm.mean(axis=1) if nm.ndim == 2 else nm
        n = int(node_imp.shape[0])
        hard = np.where(node_imp >= gs_threshold)[0]
        if len(hard) > 0:
            sel = hard
        else:
            k = max(1, int(round(gs_top_frac * n)))
            sel = np.argsort(-node_imp)[:k]
        gs_nodes = [int(x) for x in sel]
        bstrs = getattr(graph, 'building_ids_str', [''] * exp["num_nodes"])
        gs_ids = [bstrs[i] if i < len(bstrs) else '' for i in gs_nodes]
    return {
        "F_S": fs_feats,
        "F_S_full": fs_full,
        "G_S_node_indices": gs_nodes,
        "G_S_building_ids": gs_ids,
    }


# ============================================================
#  主流程
# ============================================================
def main():
    # ---- 配置驱动 ----
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    graph_dir = cfg['training_paths']['graph_data_dir']
    GRAPH_PATH = os.path.join(graph_dir, 'graph_dataset.pt')
    META_PATH = os.path.join(graph_dir, 'metadata.pkl')
    experiment_dir = cfg['training_paths'].get('experiment_dir', 'results/Experiments')
    if os.environ.get('PIPELINE_RUN_DIR'):
        experiment_dir = os.path.join(os.environ['PIPELINE_RUN_DIR'], '02_Experiments')
    CHECKPOINT_PATH = find_latest_checkpoint(experiment_dir)
    if CHECKPOINT_PATH is None:
        raise RuntimeError(f"在 {experiment_dir} 下未找到任何 best_model.pth，请先完成训练。")

    expl_cfg = cfg.get('explainer', {})
    TOPK = int(expl_cfg.get('top_k', 4))
    MAX_SAMPLES_PER_CLASS = int(expl_cfg.get('max_samples_per_class', 100))
    EXPLAIN_EPOCHS = int(expl_cfg.get('explain_epochs', 50))
    FS_THRESHOLD = float(expl_cfg.get('fs_threshold', 0.950))   # 单地块 F_S 的特征重要性阈值（参照研究 0.950）
    GS_THRESHOLD = float(expl_cfg.get('gs_threshold', 0.900))   # G_S 节点重要性硬阈值（仅作可选硬截断）
    EDGE_MASK_TYPE = None                                        # 固定走 node_mask：SAGPooling 与 GNNExplainer 的 edge_mask 不兼容（PyG2.8 下必崩）
    GS_TOP_FRAC = float(expl_cfg.get('gs_top_frac', 0.25))      # G_S 回退：节点重要性前若干比例作为核心建筑
    POOL_RATIO = float(cfg['model'].get('pool_ratio', 0.3))
    DROPOUT = float(cfg['model'].get('dropout', 0.5))
    RUN_DIR = os.environ.get('PIPELINE_RUN_DIR')
    OUTPUT_BASE_DIR = os.path.join(RUN_DIR, '03_Explainer') if RUN_DIR else r"results/Explainer_Importance"

    print(f"图数据: {GRAPH_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"TOPK(每类核心特征数)={TOPK} | MAX_SAMPLES_PER_CLASS={MAX_SAMPLES_PER_CLASS} | EXPLAIN_EPOCHS={EXPLAIN_EPOCHS}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"ep_ipt{timestamp}")
    ensure_dir(output_dir)
    device = torch.device(DEVICE)

    dataset = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    meta = load_pickle(META_PATH)
    ckpt_obj = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = get_state_dict(ckpt_obj)

    input_dim_ckpt, hidden_dim_ckpt, output_dim_ckpt = infer_checkpoint_dims(state_dict)
    feature_names = get_feature_names(meta, input_dim_ckpt)
    label_names = get_label_names(meta, output_dim_ckpt)

    model = HierarchicalGNN(
        input_dim=input_dim_ckpt,
        hidden_dim=hidden_dim_ckpt,
        output_dim=output_dim_ckpt,
        pool_ratio=POOL_RATIO,
        dropout=DROPOUT,
        edge_weight_mode=cfg['model'].get('edge_weight', 'inverse'),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    clear_explain_state(model)

    class_feature_scores = defaultdict(list)
    class_total_count = defaultdict(int)
    class_correct_count = defaultdict(int)
    class_attempt_count = defaultdict(int)
    failures = []
    gnnexplainer_blocks = []   # 每个正确预测地块的 F_S(关键特征) + G_S(关键建筑) 记录
    total_graphs = len(dataset)
    correct_count = 0
    per_class_used = defaultdict(int)

    for graph_idx, g in enumerate(dataset):
        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        true = int(g.y.item())
        class_total_count[true] += 1

        centroids = getattr(g, 'centroids', None)
        if centroids is not None:
            centroids = centroids.to(device)
        clear_explain_state(model)
        with torch.no_grad():
            if centroids is not None:
                logits = model(x, edge_index, batch, centroids)
            else:
                logits = model(x, edge_index, batch)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        clear_explain_state(model)

        if pred != true:
            continue

        if per_class_used[true] >= MAX_SAMPLES_PER_CLASS:
            class_correct_count[true] += 1
            correct_count += 1
            continue

        class_correct_count[true] += 1
        correct_count += 1
        class_attempt_count[true] += 1

        try:
            exp = compute_explainer_feature_importance(model, g, device, EXPLAIN_EPOCHS, EDGE_MASK_TYPE)
            feat_scores = exp["feat_scores"]
            class_feature_scores[true].append(feat_scores)
            per_class_used[true] += 1

            # ---- 提取 F_S（关键形态指标）与 G_S（核心子图关键建筑）----
            fs_gs = extract_fs_gs(exp, g, feature_names, FS_THRESHOLD, GS_THRESHOLD,
                                  top_k=TOPK, gs_top_frac=GS_TOP_FRAC)
            gnnexplainer_blocks.append({
                "graph_index": int(graph_idx),
                "block_id": getattr(g, 'block_id', -1),
                "true_class": int(true),
                "true_class_name": label_names[true] if true < len(label_names) else str(true),
                "pred_class": int(pred),
                "num_buildings": int(exp["num_nodes"]),
                "F_S": fs_gs["F_S"],                       # 关键形态指标（特征重要性≥阈值）
                "F_S_full": fs_gs["F_S_full"],             # 全部 22 维特征重要性
                "G_S_node_indices": fs_gs["G_S_node_indices"],  # 核心子图建筑节点索引
                "G_S_building_ids": fs_gs["G_S_building_ids"],  # 核心子图建筑 ID（用于与⑤ join）
            })
        except Exception as e:
            failures.append({
                "graph_index": int(graph_idx),
                "class_id": int(true),
                "class_name": label_names[true] if true < len(label_names) else str(true),
                "error": str(e),
            })
            clear_explain_state(model)

    # ---- 诊断：无论成败，先把失败详情落盘（原逻辑在 raise 之后才写，导致信息丢失）----
    fail_json = os.path.join(output_dir, "explainer_failures.json")
    with open(fail_json, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    # 汇总失败错误类型（取出现次数最多的前 5 种）
    err_counter = {}
    for frec in failures:
        msg = frec.get("error", "")
        err_counter[msg] = err_counter.get(msg, 0) + 1
    err_summary = sorted(err_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]

    class_ids = sorted(class_feature_scores.keys())
    if not class_ids:
        print("=" * 64)
        print("步骤④ 诊断：解释器未得到任何可用类别")
        print(f"  总图数={total_graphs}  预测正确数={correct_count}  解释失败数={len(failures)}")
        if correct_count == 0:
            print("  → 没有任何图被模型预测正确：模型在该数据集上几乎未学到")
            print("    （checkpoint 可能过弱，或图数据与训练不一致）。请先确认步骤③的")
            print("    交叉验证准确率是否合理（若仅 ~1/5 随机水平，需重训/调参）。")
        elif len(failures) > 0:
            print("  → 存在预测正确的图，但解释器对每个都抛异常。最常见的错误：")
            for msg, cnt in err_summary:
                print(f"     [{cnt} 次] {msg}")
            print("  → 若异常与 SAGPooling 的 edge_mask 相关：本框架已固定用 node_mask")
            print("    解释（SAGPooling 与 edge_mask 不兼容），无需额外配置。")
        else:
            print("  → 预测正确且解释未报错，但最终未留下任何类别记录：")
            print("    很可能 F_S/G_S 阈值(fs_threshold/gs_threshold)过高，导致所有样本")
            print("    的关键特征/关键边都被过滤为空。可下调阈值重试。")
        print(f"  失败详情已保存: {fail_json}")
        print("=" * 64)
        raise RuntimeError("解释器未得到任何可用类别，详见上方诊断与 explainer_failures.json。")

    rows = []
    topk_records = []
    summary = {
        "total_graphs": int(total_graphs),
        "correct_graphs_seen": int(correct_count),
        "top_k": TOPK,
        "all_class_statistics": [],
        "classes_included": [],
        "method": "gnnexplainer_feature_level_mean_node_mask",
        "note": "仅基于预测正确样本；每类最多抽样固定数量样本，使用 explanation.node_mask 按节点平均后得到特征重要性。",
        "checkpoint_loaded_strictly": True,
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_output_dim": int(output_dim_ckpt),
        "checkpoint_hidden_dim": int(hidden_dim_ckpt),
        "pool_ratio": float(POOL_RATIO),
        "dropout": float(DROPOUT),
        "explainer_epochs": int(EXPLAIN_EPOCHS),
        "max_samples_per_class": int(MAX_SAMPLES_PER_CLASS),
        "failure_count": int(len(failures)),
    }

    for cid in range(output_dim_ckpt):
        summary["all_class_statistics"].append({
            "class_id": int(cid),
            "class_name": label_names[cid] if cid < len(label_names) else str(cid),
            "total_graph_count": int(class_total_count.get(cid, 0)),
            "correct_graph_count": int(class_correct_count.get(cid, 0)),
            "attempted_explained_count": int(class_attempt_count.get(cid, 0)),
            "successful_explained_count": int(len(class_feature_scores.get(cid, []))),
        })

    for class_id in class_ids:
        class_name = label_names[class_id] if class_id < len(label_names) else str(class_id)
        mean_scores = np.mean(np.stack(class_feature_scores[class_id], axis=0), axis=0)
        row = {"class_id": int(class_id), "class_name": class_name}
        for feat_name, score in zip(feature_names, mean_scores):
            row[feat_name] = float(score)
        rows.append(row)

        order = np.argsort(-mean_scores)
        top_feats = []
        for rank, idx in enumerate(order[:TOPK], start=1):
            rec = {
                "class_id": int(class_id),
                "class_name": class_name,
                "rank": int(rank),
                "feature_index": int(idx),
                "feature_name": feature_names[idx],
                "feature_name_cn": FEATURE_NAME_CN.get(feature_names[idx], feature_names[idx]),
                "importance": float(mean_scores[idx]),
            }
            topk_records.append(rec)
            top_feats.append(rec)

        summary["classes_included"].append({
            "class_id": int(class_id),
            "class_name": class_name,
            "successful_explained_count": int(len(class_feature_scores[class_id])),
            "top_features": top_feats,
        })

    wide_df = pd.DataFrame(rows)
    heatmap_df = wide_df.set_index("class_name")[feature_names]
    norm_df = normalize_rows(heatmap_df)

    wide_csv = os.path.join(output_dir, "feature_importance_by_class.csv")
    norm_csv = os.path.join(output_dir, "feature_importance_by_class_normalized.csv")
    topk_csv = os.path.join(output_dir, "feature_importance_topk.csv")
    summary_json = os.path.join(output_dir, "feature_importance_summary.json")
    fail_json = os.path.join(output_dir, "explainer_failures.json")
    heatmap_png = os.path.join(output_dir, "feature_importance_heatmap.png")

    wide_df.to_csv(wide_csv, index=False, encoding="utf-8-sig")
    norm_df.to_csv(norm_csv, encoding="utf-8-sig")
    pd.DataFrame(topk_records).to_csv(topk_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(fail_json, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    # ---- ④ 新增：落盘 G_S(核心子图关键建筑) + F_S(关键形态指标)，供步骤⑥回贴定地块构型 ----
    core_path = os.path.join(output_dir, "gnnexplainer_core.json")
    core_out = {
        "method": "gnnexplainer_fs_gs",
        "thresholds": {"feature_fs": FS_THRESHOLD, "gs_node": GS_THRESHOLD,
                       "gs_top_frac": GS_TOP_FRAC, "edge_mask_type": EDGE_MASK_TYPE},
        "feature_names": list(feature_names),
        "num_blocks_explained": len(gnnexplainer_blocks),
        "blocks": gnnexplainer_blocks,
    }
    with open(core_path, "w", encoding="utf-8") as f:
        json.dump(core_out, f, ensure_ascii=False, indent=2)
    print(f"[完成] G_S/F_S 核心提取已保存: {core_path}（{len(gnnexplainer_blocks)} 个地块）")

    # ---- ④ 新增：聚合全局 F_S（跨所有被解释地块的节点重要性均值）----
    # 这就是供步骤⑤聚类用的「核心城市形态特征集」特征子空间（fs_from_explainer 模式）。
    if gnnexplainer_blocks:
        fi_acc = {f: 0.0 for f in feature_names}
        fi_cnt = {f: 0 for f in feature_names}
        for blk in gnnexplainer_blocks:
            fsf = blk.get("F_S_full", {}) or {}
            for f, v in fsf.items():
                if f in fi_acc:
                    fi_acc[f] += float(v); fi_cnt[f] += 1
        global_imp = {f: (fi_acc[f] / fi_cnt[f] if fi_cnt[f] > 0 else 0.0) for f in feature_names}
        sorted_imp = sorted(global_imp.items(), key=lambda kv: kv[1], reverse=True)
        # 把每个功能类别各自的 Top-K 关键特征合并去重，作为「核心城市形态特征集」。
        # 数量由数据自然产生（上限 = 类别数 × top_k，去重后更少），不做固定数量截断。
        # 依据：参照研究的 10 个核心形态指标本就是各功能类别关键特征的并集，
        # 「10」是结果而非预设；按全局重要性硬取前 N 个会漏掉只对单一功能类别
        # 重要、但全局均值排名靠后的判别性特征。
        union, seen = [], set()
        for rec in topk_records:          # 已按 class_id -> rank 顺序生成
            fname = rec["feature_name"]
            if fname not in seen:
                seen.add(fname)
                union.append(fname)
        selected = union
        if not selected:  # 兜底：至少保留 top-5，避免聚类无特征可用
            selected = [f for f, _ in sorted_imp[:5]]
        # 按全局重要性排序输出，便于阅读（不改变集合内容）
        _rank = {f: i for i, (f, _) in enumerate(sorted_imp)}
        selected = sorted(selected, key=lambda f: _rank.get(f, 10 ** 6))
        fs_global = {
            "method": "gnnexplainer_global_feature_importance_mean",
            "selection_mode": "union_topk_per_class",
            "per_class_top_k": TOPK,
            "per_block_fs_threshold": FS_THRESHOLD,
            "num_selected": len(selected),
            "num_blocks_used": len(gnnexplainer_blocks),
            "global_importance": {f: round(v, 6) for f, v in sorted_imp},
            "selected_features": selected,
            "note": ("供步骤⑤聚类特征子空间（clustering.feature_source=fs_from_explainer 时使用）。"
                     "= 各功能类别 Top-K 关键特征的并集去重（不截断），数量由数据自然产生。"),
        }
        fs_global_path = os.path.join(output_dir, "fs_global.json")
        with open(fs_global_path, "w", encoding="utf-8") as f:
            json.dump(fs_global, f, ensure_ascii=False, indent=2)
        _src = f"各类 Top-{TOPK} 并集（不截断，{len(class_ids)} 个类别）"
        print(f"[完成] 全局 F_S（核心城市形态特征集）已保存: {fs_global_path}（{len(selected)} 个特征｜选法: {_src}）")
        print(f"    选中特征: {selected}")

    save_heatmap_data(norm_df, feature_names, label_names, output_dir)
    plot_heatmap_with_cn_labels(norm_df, feature_names, heatmap_png, dpi=600)

    print(f"输出目录: {output_dir}")
    print(f"总图数量: {total_graphs}")
    print(f"预测正确样本数: {correct_count}")
    print(f"热力图类别数: {len(class_ids)}")
    print(f"checkpoint 输出类别数: {output_dim_ckpt}")
    print("模型严格加载: True")
    print(f"解释失败样本数: {len(failures)}")
    print(f"热力图: {heatmap_png}")
    print("各类别核心形态特征 (Top-K):")
    for rec in topk_records:
        print(f"  [{rec['class_name']}] #{rec['rank']} {rec['feature_name_cn']} ({rec['feature_name']}) = {rec['importance']:.4f}")


if __name__ == "__main__":
    main()
