"""
GNN_Train.py
层次化图神经网络训练代码（优化实验版 / opt）

相对 3GNN_train_v2.py 的改进：
  [P0] 统一配置文件：读取 config.yaml；建图脚本会自动写入正确的 graph_data_dir，
       彻底修复旧版“建图更新 A 配置、训练读 B 配置”导致静默训旧数据的 bug。
  [P1] 边权：GCN 模式下，按论文用建筑间地理距离作边权 W（edge_weight = 1/距离，可配置 raw/inverse/none）。
       距离在模型内逐层用节点质心实时计算（SAGPooling 后节点子集随之重算），更准确。
  [⑧ ] 双模型对比：model.type = gcn | gat。GAT 用注意力替代固定距离权重，可直接对比两种建模思路。
  [修复] docstring 维度 23 → 22；保持“地块=图、建筑=节点、功能=地块级标签”（与 EULUC2.0 一致，不改 building 级）。

注意：本版本不使用类别权重，与论文策略一致。
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
import pickle
import datetime
import warnings
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import numpy_pickle_shim  # 兼容 numpy2.x 保存的 pkl（numpy._core 别名到 core）
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_recall_fscore_support, classification_report
)

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool, global_max_pool, JumpingKnowledge
from matplotlib import rcParams

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['font.serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

CLASS_ID_TO_CN = {
    1.0: "居住",
    2.0: "商业",
    3.0: "工业",
    4.0: "交通",
    5.0: "公共管理与服务",
}

def set_seed(seed: int = 42):
    """固定随机种子，保证实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HierarchicalGNN(nn.Module):
    """
    层次化图神经网络（优化实验版 / opt）

    对标论文 Fig. 3，并做以下改进：
      - 3层图卷积 + 3层 SAGPooling（Conv-Pool 交替）
      - 每层卷积后增加 BatchNorm1d（稳定训练，加速收敛）
      - 每层池化后进行 Concat Pooling (mean + max)
      - 三层图级表示相加（Jump Knowledge / Skip Connection）
      - 3层 MLP 分类器（含 BatchNorm + Dropout）
      - [P1] 按真实地理距离算边权（从 centroids 逐层重算；edge_weight=none 时关闭）

    Parameters
    ----------
    input_dim  : 节点特征维度（22）
    hidden_dim : 隐藏层维度（默认 128）
    output_dim : 分类数
    pool_ratio : 每层保留的节点比例（0.30）
    dropout    : Dropout 比例（0.5）
    edge_weight_mode : 'inverse' | 'raw' | 'none'
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 pool_ratio=0.3, dropout=0.5, edge_weight_mode='inverse'):
        super(HierarchicalGNN, self).__init__()
        self.edge_weight_mode = edge_weight_mode
        self.dropout_rate = dropout

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # BatchNorm 层（稳定训练过程）
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # 池化层：SAGPooling（全局 top-k 选择）
        self.pool1 = SAGPooling(hidden_dim, ratio=pool_ratio)
        self.pool2 = SAGPooling(hidden_dim, ratio=pool_ratio)
        self.pool3 = SAGPooling(hidden_dim, ratio=pool_ratio)

        # Jump Knowledge: 拼接三层 graph-level 表示后投影回原维度
        # 替换原 "out1 + out2 + out3" 直接相加；cat 保留三层全部通道，再
        # 用 1x1 式线性投影降回 hidden_dim*2，与 JK-Net 标准做法一致。
        self.jk = JumpingKnowledge(mode='cat')
        self.jk_proj = nn.Linear(3 * hidden_dim * 2, hidden_dim * 2)

        # MLP 分类器（输入为 hidden_dim * 2，因为 concat mean+max pooling）
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
        """[P1] 按真实地理距离计算边权。"""
        if self.edge_weight_mode == 'none' or centroids is None:
            return None
        if edge_index.numel() == 0:
            return None
        row, col = edge_index
        dist = torch.norm(centroids[row] - centroids[col], dim=1) + 1e-6
        if self.edge_weight_mode == 'inverse':
            return 1.0 / dist          # 越近越强
        return dist                     # raw = 距离本身

    def _readout(self, x, batch):
        """Concat Pooling: mean + max"""
        return torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=1)

    def forward(self, x, edge_index, batch, centroids=None):
        # Layer 1: Conv + BN + ReLU + Dropout + Pool
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

        # Layer 2: Conv + BN + ReLU + Dropout + Pool
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

        # Layer 3: Conv + BN + ReLU + Dropout + Pool
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

        # Jump Knowledge: 拼接三层 graph-level 表示（替换原直接相加）
        out = self.jk([out1, out2, out3])   # (B, 3 * hidden_dim * 2)
        out = self.jk_proj(out)             # (B, hidden_dim * 2)
        return self.mlp(out)


class GNNModelTrainer:
    """GNN 模型训练器（10折分层交叉验证，opt 版）"""

    def __init__(self, config_path="config.yaml", config=None):
        if config is not None:
            self.config = config
        else:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"找不到配置文件: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

        self.seed = self.config.get('training', {}).get('seed', 42)
        set_seed(self.seed)

        use_gpu = self.config['device'].get('use_gpu', True)
        force_cpu = self.config['device'].get('force_cpu', False)
        self.device = torch.device(
            'cuda' if (use_gpu and torch.cuda.is_available() and not force_cpu) else 'cpu'
        )
        print(f"计算设备: {self.device}")
        print(f"随机种子: {self.seed}")

        cfg_m = self.config['model']
        self.edge_weight_mode = cfg_m.get('edge_weight', 'inverse')
        print(f"图卷积: GCN  |  边权模式: {self.edge_weight_mode}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.config['training_paths'].get('experiment_dir', 'results/Experiments')
        if os.environ.get('PIPELINE_RUN_DIR'):
            exp_dir = os.path.join(os.environ['PIPELINE_RUN_DIR'], '02_Experiments')
        model_tag = f"gcn_{self.edge_weight_mode}"
        self.output_dir = os.path.join(exp_dir, f"GNN_{model_tag}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"实验结果将保存至: {self.output_dir}")

        self.dataset = None
        self.label_encoder = None
        self.feature_names = None

    def load_data(self):
        data_dir = self.config['training_paths']['graph_data_dir']
        dataset_path = os.path.join(data_dir, 'graph_dataset.pt')
        meta_path = os.path.join(data_dir, 'metadata.pkl')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"找不到图数据文件: {dataset_path}（请先运行 Build_Graphs.py）")

        print(f"\n加载图数据: {dataset_path}")
        self.dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.label_encoder = meta['label_encoder']
        self.feature_names = meta['feature_names']

        n_graphs = len(self.dataset)
        n_features = self.dataset[0].x.shape[1]
        labels = [d.y.item() for d in self.dataset]
        label_dist = pd.Series(labels).value_counts().sort_index()

        print(f"图数量: {n_graphs}  |  节点特征数: {n_features}")
        print("类别分布:")
        for cls_id, cnt in label_dist.items():
            cls_name = self.label_encoder.inverse_transform([cls_id])[0]
            print(f"  [{cls_id}] {cls_name}: {cnt} 个图 ({cnt/n_graphs*100:.1f}%)")

    def _train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for data in loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            centroids = getattr(data, 'centroids', None)
            out = model(data.x, data.edge_index, data.batch, centroids)
            loss = criterion(out, data.y.view(-1))
            loss.backward()
            # 梯度裁剪，防止梯度爆炸（对更大的模型尤其有用）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        return total_loss / len(loader.dataset), acc

    @torch.no_grad()
    def _eval_epoch(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for data in loader:
            data = data.to(self.device)
            centroids = getattr(data, 'centroids', None)
            out = model(data.x, data.edge_index, data.batch, centroids)
            loss = criterion(out, data.y.view(-1))

            total_loss += loss.item() * data.num_graphs
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return total_loss / len(loader.dataset), acc, macro_f1, weighted_f1, all_preds, all_labels

    def _train_one_fold(self, train_dataset, val_dataset, test_dataset, fold_idx):
        cfg_m = self.config['model']
        cfg_t = self.config['training']

        input_dim = self.dataset[0].x.shape[1]

        model = HierarchicalGNN(
            input_dim=input_dim,
            hidden_dim=cfg_m.get('hidden_dim', 128),
            output_dim=cfg_m['output_dim'],
            pool_ratio=cfg_m.get('pool_ratio', 0.3),
            dropout=cfg_m.get('dropout', 0.5),
            edge_weight_mode=cfg_m.get('edge_weight', 'inverse'),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg_t['learning_rate'],
            weight_decay=cfg_t['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=30,
            min_lr=1e-6
        )  # [修复] 去掉 verbose 参数：PyTorch 2.x 已移除该参数

        label_smoothing = cfg_t.get('label_smoothing', 0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        batch_size = cfg_t['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_val_wf1 = -1.0
        best_state = None
        patience_counter = 0
        train_losses, val_losses = [], []

        epochs = cfg_t['epochs']
        patience = cfg_t.get('patience', 100)

        pbar = tqdm(
            range(epochs),
            desc=f"Fold {fold_idx}",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for epoch in pbar:
            tr_loss, tr_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_macro, val_wf1, _, _ = self._eval_epoch(model, val_loader, criterion)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'tr_loss': f'{tr_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.4f}',
                'val_wf1': f'{val_wf1:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_wf1 = val_wf1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                pbar.close()
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        else:
            pbar.close()

        model.load_state_dict(best_state)
        test_loss, test_acc, test_macro, test_wf1, preds, trues = self._eval_epoch(model, test_loader, criterion)

        return {
            'model_state_dict': copy.deepcopy(best_state),
            'best_val_loss': best_val_loss,
            'best_val_weighted_f1': best_val_wf1,
            'test_acc': test_acc,
            'test_macro_f1': test_macro,
            'test_weighted_f1': test_wf1,
            'preds': preds,
            'trues': trues,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

    def cross_validate(self):
        n_splits = self.config['training']['n_splits']
        quick_fold = self.config['training'].get('quick_fold', False)
        print("\n" + "=" * 70)
        mode_desc = "单折快速验证 (quick_fold)" if quick_fold else f"{n_splits} 折分层交叉验证"
        print(f"开始 {mode_desc} (opt) | 图卷积=GCN 边权={self.edge_weight_mode}")
        print("      BatchNorm in GCN layers, Gradient Clipping, patience=100")
        print("=" * 70)

        self.load_data()

        labels = [d.y.item() for d in self.dataset]

        fold_results = []
        all_fold_true = []
        all_fold_pred = []
        best_model_state = None
        best_model_fold = None
        best_model_val_loss = float('inf')

        if quick_fold:
            # 单折快速验证：一次分层抽样 80/10/10，用于改模型后分钟级确认"能跑通、loss 在降"
            from sklearn.model_selection import train_test_split
            idx_all = np.arange(len(labels))
            trval_idx, test_idx = train_test_split(
                idx_all, test_size=0.1, stratify=labels, random_state=self.seed)
            trval_labels = [labels[i] for i in trval_idx]
            train_idx, val_idx = train_test_split(
                trval_idx, test_size=0.1111, stratify=trval_labels, random_state=self.seed)
            fold_splits = [(train_idx, val_idx, test_idx)]
            n_folds = 1
            print("  [quick_fold=true] 单折快速验证模式 (80/10/10 分层抽样)")
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            fold_splits = []
            for train_val_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
                train_val_labels = [labels[i] for i in train_val_idx]
                inner_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=self.seed)
                inner_train_idx, inner_val_idx = next(
                    inner_skf.split(np.zeros(len(train_val_labels)), train_val_labels)
                )
                train_idx = [train_val_idx[i] for i in inner_train_idx]
                val_idx = [train_val_idx[i] for i in inner_val_idx]
                fold_splits.append((train_idx, val_idx, test_idx))
            n_folds = n_splits

        for fold in range(n_folds):
            train_idx, val_idx, test_idx = fold_splits[fold]
            print(f"\n{'─' * 50}")
            print(f"  Fold {fold + 1} / {n_folds}")
            print(f"{'─' * 50}")

            train_ds = [self.dataset[i] for i in train_idx]
            val_ds = [self.dataset[i] for i in val_idx]
            test_ds = [self.dataset[i] for i in test_idx]

            print(f"  训练集: {len(train_ds)}  验证集: {len(val_ds)}  测试集: {len(test_ds)}")

            result = self._train_one_fold(train_ds, val_ds, test_ds, fold + 1)

            print(
                f"  Fold {fold + 1} 结果: "
                f"ValLoss={result['best_val_loss']:.4f}  "
                f"ValWeightedF1={result['best_val_weighted_f1']:.4f}  "
                f"TestAcc={result['test_acc']:.4f}  "
                f"TestMacro-F1={result['test_macro_f1']:.4f}  "
                f"TestWeighted-F1={result['test_weighted_f1']:.4f}"
            )

            fold_results.append({
                'fold': fold + 1,
                'best_val_loss': result['best_val_loss'],
                'best_val_weighted_f1': result['best_val_weighted_f1'],
                'accuracy': result['test_acc'],
                'macro_f1': result['test_macro_f1'],
                'weighted_f1': result['test_weighted_f1']
            })

            all_fold_true.extend(result['trues'])
            all_fold_pred.extend(result['preds'])

            if result['best_val_loss'] < best_model_val_loss:
                best_model_val_loss = result['best_val_loss']
                best_model_state = copy.deepcopy(result['model_state_dict'])
                best_model_fold = fold + 1

            self._plot_training_curve(result['train_losses'], result['val_losses'], fold + 1)

        print("\n" + "=" * 70)
        print(f"交叉验证最终结果 (Aggregate Metrics across {n_folds} fold(s))")
        print("=" * 70)

        agg_acc = accuracy_score(all_fold_true, all_fold_pred)
        agg_macro = f1_score(all_fold_true, all_fold_pred, average='macro', zero_division=0)
        agg_weighted = f1_score(all_fold_true, all_fold_pred, average='weighted', zero_division=0)
        agg_micro = f1_score(all_fold_true, all_fold_pred, average='micro', zero_division=0)

        prec_macro, rec_macro, _, _ = precision_recall_fscore_support(
            all_fold_true, all_fold_pred, average='macro', zero_division=0)
        prec_weighted, rec_weighted, _, _ = precision_recall_fscore_support(
            all_fold_true, all_fold_pred, average='weighted', zero_division=0)

        print(f"Overall Accuracy:       {agg_acc:.4f}")
        print(f"Micro F1-score:         {agg_micro:.4f}")
        print(f"Macro F1-score:         {agg_macro:.4f}")
        print(f"Weighted F1-score:      {agg_weighted:.4f}")
        print(f"Macro Precision:        {prec_macro:.4f}")
        print(f"Weighted Precision:     {prec_weighted:.4f}")
        print(f"Macro Recall:           {rec_macro:.4f}")
        print(f"Weighted Recall:        {rec_weighted:.4f}")

        fold_df = pd.DataFrame(fold_results)
        print("\n各折统计 (Mean ± SD):")
        for col in ['accuracy', 'macro_f1', 'weighted_f1']:
            mean_v = fold_df[col].mean()
            sd_v = fold_df[col].std() if len(fold_df) > 1 else 0.0  # 单折无标准差
            print(f"  {col}: {mean_v:.4f} ± {sd_v:.4f}")

        print("\n各类别详细指标:")
        n_classes = len(self.label_encoder.classes_)
        print(classification_report(
            all_fold_true, all_fold_pred,
            labels=list(range(n_classes)),           # 强制覆盖全部类别，避免单折缺类导致长度不匹配
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))

        # [P0] 先持久化重要产物（模型权重/指标 csv/json），再做展示性绘图与打印。
        # 避免后续 print 因 GBK 编码问题崩溃时，已训练好的模型/指标反而丢失（曾白跑 4 小时）。
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(self.output_dir, 'best_model.pth'))
            print(
                f"\n[完成] 最优模型已保存（依据验证集 loss，来自 Fold {best_model_fold}，"
                f"best_val_loss={best_model_val_loss:.4f}）"
            )

        fold_df.to_csv(
            os.path.join(self.output_dir, 'fold_results.csv'),
            index=False, encoding='utf-8-sig'
        )

        summary = {
            'version': 'opt',
            'model_type': 'gcn',
            'edge_weight_mode': self.edge_weight_mode,
            'aggregate_metrics': {
                'overall_accuracy': float(agg_acc),
                'micro_f1': float(agg_micro),
                'macro_f1': float(agg_macro),
                'weighted_f1': float(agg_weighted),
                'macro_precision': float(prec_macro),
                'weighted_precision': float(prec_weighted),
                'macro_recall': float(rec_macro),
                'weighted_recall': float(rec_weighted)
            },
            'fold_mean_std': {
                col: {'mean': float(fold_df[col].mean()), 'std': float(fold_df[col].std())}
                for col in ['accuracy', 'macro_f1', 'weighted_f1']
            },
            'best_model_selection': {
                'criterion': 'lowest_validation_loss',
                'fold': best_model_fold,
                'best_val_loss': float(best_model_val_loss) if best_model_val_loss != float('inf') else None
            },
            'fold_results': fold_results,
            'config': self.config
        }
        with open(os.path.join(self.output_dir, 'cv_results.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        # 展示性绘图放最后：内部 save 已完成；即便后续打印异常也不影响上述已落盘产物
        self._plot_confusion_matrix(all_fold_true, all_fold_pred)

        print(f"\n所有结果已保存至: {self.output_dir}")
        return summary

    def _plot_training_curve(self, train_losses, val_losses, fold_idx):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, label='训练损失（Train Loss）', linewidth=1.5)
        ax.plot(val_losses, label='验证损失（Val Loss）', linewidth=1.5)
        ax.set_xlabel('训练轮次（Epoch）')
        ax.set_ylabel('损失值（Loss）')
        ax.set_title(f'第{fold_idx}折 - 训练过程收敛曲线')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'training_curve_fold{fold_idx}.png'), dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, trues, preds):

        import pickle
        import numpy as np
        import pandas as pd

        cm = confusion_matrix(trues, preds, labels=list(range(len(self.label_encoder.classes_))))

        # ========== 第一步：保存原始数据 ==========
        cm_data = {
            'confusion_matrix': cm,
            'true_labels': np.array(trues),
            'pred_labels': np.array(preds),
            'class_names': list(self.label_encoder.classes_),
            'class_mapping': CLASS_ID_TO_CN
        }

        cm_pickle_path = os.path.join(self.output_dir, 'confusion_matrix_data.pkl')
        with open(cm_pickle_path, 'wb') as f:
            pickle.dump(cm_data, f)
        print(f"[完成] 混淆矩阵原始数据已保存: {cm_pickle_path}")

        # 保存为CSV格式（便于查看）
        cm_df = pd.DataFrame(
            cm,
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        cm_csv_path = os.path.join(self.output_dir, 'confusion_matrix_data.csv')
        cm_df.to_csv(cm_csv_path, encoding='utf-8-sig')
        print(f"[完成] 混淆矩阵CSV已保存: {cm_csv_path}")

        # ========== 第二步：计算各类准确率 ==========
        class_accuracies = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            if cm[i].sum() > 0:
                accuracy = cm[i, i] / cm[i].sum()
                class_accuracies[class_name] = accuracy
            else:
                class_accuracies[class_name] = 0.0

        # ========== 第三步：绘制混淆矩阵（中文标签） ==========
        class_names_cn = [CLASS_ID_TO_CN.get(float(c), c) for c in self.label_encoder.classes_]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_cn,
            yticklabels=class_names_cn,
            ax=ax,
            cbar_kws={'label': '样本数'}
        )
        ax.set_xlabel('预测类别', fontsize=13, fontweight='bold')
        ax.set_ylabel('真实类别', fontsize=13, fontweight='bold')
        ax.set_title('图X 混淆矩阵（10折交叉验证聚合结果）', fontsize=15, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)

        plt.tight_layout()
        cm_png_path = os.path.join(self.output_dir, 'aggregate_confusion_matrix.png')
        plt.savefig(cm_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[完成] 混淆矩阵图已保存: {cm_png_path}")

        # ========== 第四步：打印各类准确率 ==========
        print("\n" + "=" * 60)
        print("各类别准确率统计")
        print("=" * 60)
        for class_name in self.label_encoder.classes_:
            class_name_cn = CLASS_ID_TO_CN.get(float(class_name), class_name)
            accuracy = class_accuracies[class_name]
            print(f"  {class_name_cn:12} ({class_name}): {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("=" * 60 + "\n")

        accuracy_json_path = os.path.join(self.output_dir, 'class_accuracies.json')
        accuracy_data = {
            CLASS_ID_TO_CN.get(float(k), k): float(v)
            for k, v in class_accuracies.items()
        }
        with open(accuracy_json_path, 'w', encoding='utf-8') as f:
            json.dump(accuracy_data, f, ensure_ascii=False, indent=2)
        print(f"[完成] 各类准确率已保存: {accuracy_json_path}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    trainer = GNNModelTrainer(config_path=cfg_path)
    trainer.cross_validate()
