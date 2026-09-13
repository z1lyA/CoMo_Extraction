# =============================================================================
# cross_city_transfer.py  —— 跨城市泛化推理（用源城市训练好的 GNN 直接推断目标城市）
# -----------------------------------------------------------------------------
# 目的：把「源城市训练好的模型」直接套用到「目标城市」的数据上，验证跨城市泛化性。
#
# 核心难点与处理方式：
#   1) 特征空间对齐：Build_Graphs 在存图时会用【各城市自己的 StandardScaler】缩放
#      节点特征。源模型是在「源城缩放空间」训的，目标城图若在「目标城缩放空间」直接喂
#      会错位。处理：先 tgt_scaler.inverse_transform 还原原始特征，再
#      src_scaler.transform 投到源城缩放空间（两个 scaler 都在各自的 metadata.pkl 里）。
#   2) 标签空间对齐：各城市用同一套 EULUC Level1 标签（居住/商业/工业/交通/公服），
#      但 LabelEncoder 的索引可能不同。处理：用源城市 label_encoder 把「真实标签字符串」
#      和「预测索引」统一到源城市索引空间再比对；目标城市出现源城市没有的类别会被标
#      记为 -1 并跳过（跨城市标签体系不一致时才会触发）。
#
# 用法：
#   # (A) 自测：源=目标（同一城市），应复现该城精度，用来验证本脚本逻辑正确
#   python cross_city_transfer.py \
#       --src_config config.yaml \
#       --src_graph_dir results/<源城图目录>/graphs_xxxxxx \
#       --src_model results/<源城实验目录>/GNN_gcn_none_xxxxxx/best_model.pth \
#       --tgt_graph_dir results/<源城图目录>/graphs_xxxxxx \
#       --out results/transfer_self_test
#
#   # (B) 真实跨城市（源城模型 -> 目标城）：
#   #   1) 先构建目标城图（准备好目标城 data/*.shp 后跑流水线/建图），
#   #      记下它打印的 graphs_YYYYMMDD_HHMMSS 目录，传给 --tgt_graph_dir
#   #   2) 运行本脚本：
#   python cross_city_transfer.py \
#       --src_config config.yaml \
#       --src_graph_dir results/<源城图目录>/graphs_xxxxxx \
#       --src_model results/<源城实验目录>/GNN_gcn_none_xxxxxx/best_model.pth \
#       --tgt_graph_dir <目标城 graphs_xxxxxx 目录> \
#       --out results/transfer_src2tgt
#
#   注：源模型请用你自己训练好的 best_model.pth（--src_model），脚本不预设任何源城市。
#
# 产物（写到 --out 目录）：
#   transfer_metrics.json      —— 总体/逐类精度、macro/weighted F1
#   transfer_predictions.csv  —— 每个区块(block)的真实/预测标签与概率
#   transfer_predictions.shp  —— 区块级预测图（供 WebGIS 热力图直接使用）
# =============================================================================
import argparse
import os
import sys
import json
import time
import pickle

import numpy as np
import pandas as pd
import torch

# 项目内复用：从 src/ 包导入模型与建图逻辑（避免重复实现导致架构不一致）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)                              # 让 src 作为包被加载
from src.GNN_Train import HierarchicalGNN
from src.Build_Graphs import UrbanMorphologyGraphBuilder

sys.stdout.reconfigure(errors='replace')
sys.stderr.reconfigure(errors='replace')


# --------------------------------------------------------------------------- #
# 工具：构建目标城市图（若目录不存在则现场构建并写出 graph_dataset.pt + metadata.pkl）
# --------------------------------------------------------------------------- #
def build_target_graphs(tgt_config_path, tgt_graph_dir):
    print(f'[构建目标城市图] {tgt_config_path} -> {tgt_graph_dir}')
    import yaml
    with open(tgt_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    builder = UrbanMorphologyGraphBuilder(config)
    buildings, landuse = builder.load_and_preprocess_data()
    dataset, stats, building_id_field = builder.process_all_blocks(buildings, landuse)
    if not dataset:
        raise RuntimeError('未能构建任何图，请检查目标城市数据与配置。')
    os.makedirs(tgt_graph_dir, exist_ok=True)
    torch.save(dataset, os.path.join(tgt_graph_dir, 'graph_dataset.pt'))
    meta = {
        'label_encoder': builder.label_encoder,
        'feature_scaler': builder.feature_scaler,
        'feature_names': builder.feature_names,
        'building_id_field': building_id_field,
        'config': config,
    }
    with open(os.path.join(tgt_graph_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'    写出图: {len(dataset)} 个 | 特征数: {dataset[0].x.shape[1]}')


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='Cross-city transfer inference')
    ap.add_argument('--src_config', required=True, help='源城市(训练模型)配置文件，用于取模型结构超参')
    ap.add_argument('--src_graph_dir', required=True, help='源城市图目录(含 graph_dataset.pt + metadata.pkl)')
    ap.add_argument('--src_model', required=True, help='源城市 best_model.pth 路径')
    ap.add_argument('--tgt_graph_dir', required=True, help='目标城市图目录(含 graph_dataset.pt + metadata.pkl)')
    ap.add_argument('--tgt_config', default=None,
                    help='目标城市配置；仅当 --tgt_graph_dir 不存在时用于现场构建目标图')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    device = torch.device('cpu')
    print(f'计算设备: {device}')

    # ---- 1) 源城市：模型 + 元数据 ----
    print('[1/5] 加载源城市模型与元数据...')
    import yaml
    with open(args.src_config, 'r', encoding='utf-8') as f:
        src_cfg = yaml.safe_load(f)
    cfg_m = src_cfg['model']

    with open(os.path.join(args.src_graph_dir, 'metadata.pkl'), 'rb') as f:
        src_meta = pickle.load(f)
    gz_scaler = src_meta['feature_scaler']
    gz_le = src_meta['label_encoder']
    gz_feature_names = list(src_meta['feature_names'])
    n_classes = len(gz_le.classes_)
    print(f'    源标签空间({n_classes}类): {list(gz_le.classes_)}')

    model = HierarchicalGNN(
        input_dim=len(gz_feature_names),
        hidden_dim=cfg_m.get('hidden_dim', 128),
        output_dim=n_classes,
        pool_ratio=cfg_m.get('pool_ratio', 0.3),
        dropout=cfg_m.get('dropout', 0.5),
        edge_weight_mode=cfg_m.get('edge_weight', 'none'),
    ).to(device)
    sd = torch.load(args.src_model, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    print(f'    模型加载完成: {args.src_model}')

    # ---- 2) 目标城市：图 + 元数据（缺失则现场构建）----
    print('[2/5] 加载目标城市图...')
    if not os.path.exists(os.path.join(args.tgt_graph_dir, 'graph_dataset.pt')):
        if not args.tgt_config:
            raise SystemExit('[错误] 目标图目录不存在，且未提供 --tgt_config 用于构建')
        build_target_graphs(args.tgt_config, args.tgt_graph_dir)

    tgt_dataset = torch.load(
        os.path.join(args.tgt_graph_dir, 'graph_dataset.pt'), map_location='cpu', weights_only=False)
    with open(os.path.join(args.tgt_graph_dir, 'metadata.pkl'), 'rb') as f:
        tgt_meta = pickle.load(f)
    tgt_scaler = tgt_meta['feature_scaler']
    tgt_le = tgt_meta['label_encoder']
    tgt_feature_names = list(tgt_meta['feature_names'])
    same_city = (args.src_graph_dir.rstrip('/\\') == args.tgt_graph_dir.rstrip('/\\'))

    # 特征名一致性：本项目的源/目标图均由同一 Build_Graphs 代码生成，特征定义按
    # 位置对齐。若仅“名字”不同（历史模型用旧名、新版已改名，如 Concavity->Solidity）
    # 属正常，按位置使用即可；若“维数”不同才是真正的特征体系不一致，必须报错。
    if len(tgt_feature_names) != len(gz_feature_names):
        raise SystemExit(
            f'[错误] 源/目标特征维数不一致 ({len(gz_feature_names)} vs {len(tgt_feature_names)})，'
            '两城市特征体系不同，无法对齐。')
    if tgt_feature_names != gz_feature_names:
        print(f'[信息] 源/目标特征名不同（常见原因：历史模型用旧名、新版已改名，'
              f'如 Concavity->Solidity），但维数相同、均由同一流水线按位置生成，'
              f'按位置对齐使用，结果有效。')
    print(f'    目标图数: {len(tgt_dataset)} | 目标特征数: {len(tgt_feature_names)}')

    # ---- 3) 逐图推理：把目标特征投到源缩放空间，forward ----
    print('[3/5] 逐图推理（特征空间对齐到源城市）...')
    preds_gz = []          # 预测在源索引空间
    trues_gz = []          # 真实标签在源索引空间（-1=源空间无此类别）
    block_ids = []
    max_probs = []

    gz_scaler_features = list(getattr(gz_scaler, 'feature_names_in_', gz_feature_names))

    with torch.no_grad():
        for g in tgt_dataset:
            x = g.x.numpy()
            # 特征空间对齐：先还原目标原始特征，再投到源缩放空间
            if not same_city:
                x_raw = tgt_scaler.inverse_transform(x)
                x = gz_scaler.transform(x_raw)
            xt = torch.tensor(x, dtype=torch.float)
            data = type(g)(
                x=xt,
                edge_index=g.edge_index,
                edge_attr=getattr(g, 'edge_attr', None),
                y=g.y,
            )
            if hasattr(g, 'batch'):
                data.batch = g.batch
            if hasattr(g, 'centroids'):
                data.centroids = g.centroids
            data = data.to(device)
            centroids = getattr(data, 'centroids', None)
            out = model(data.x, data.edge_index, data.batch, centroids)
            prob = torch.softmax(out, dim=1)
            p = prob.argmax(dim=1).item()
            mp = prob.max(dim=1).values.item()
            preds_gz.append(p)
            max_probs.append(mp)
            # 真实标签：目标 LabelEncoder -> 字符串 -> 源 LabelEncoder
            true_str = str(tgt_le.inverse_transform([g.y.item()])[0])
            try:
                t_gz = int(gz_le.transform([true_str])[0])
                trues_gz.append(t_gz)
            except ValueError:
                trues_gz.append(-1)  # 源空间没有该类别
            # block_id（从图级属性取，建图时写入）
            bid = getattr(g, 'block_id', None)
            block_ids.append(int(bid) if bid is not None else -1)

    preds_gz = np.array(preds_gz)
    trues_gz = np.array(trues_gz)
    valid = trues_gz >= 0
    if valid.sum() == 0:
        raise RuntimeError('目标城市标签与源城市标签空间无交集，无法评估（检查两城市 EULUC 类别体系）。')

    # ---- 4) 评估 ----
    print('[4/5] 评估跨城市泛化性能...')
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(trues_gz[valid], preds_gz[valid])
    macro = f1_score(trues_gz[valid], preds_gz[valid], average='macro', zero_division=0)
    weighted = f1_score(trues_gz[valid], preds_gz[valid], average='weighted', zero_division=0)
    print(f'    跨城市精度(accuracy): {acc:.4f}')
    print(f'    macro-F1: {macro:.4f} | weighted-F1: {weighted:.4f}')
    print(f'    有效评估样本: {valid.sum()} / {len(trues_gz)} '
          f'({len(trues_gz) - valid.sum()} 个目标类别不在源空间，已跳过)')
    rep = classification_report(
        trues_gz[valid], preds_gz[valid],
        labels=list(range(n_classes)),
        target_names=[str(c) for c in gz_le.classes_],
        zero_division=0, output_dict=True)

    metrics = {
        'source': os.path.basename(os.path.dirname(args.src_model)),
        'target_graph_dir': args.tgt_graph_dir,
        'same_city_self_test': bool(same_city),
        'n_graphs': int(len(tgt_dataset)),
        'n_valid_eval': int(valid.sum()),
        'n_skipped_unseen_class': int((~valid).sum()),
        'accuracy': float(acc),
        'macro_f1': float(macro),
        'weighted_f1': float(weighted),
        'source_classes': [str(c) for c in gz_le.classes_],
        'per_class': {str(gz_le.classes_[i]): rep[str(gz_le.classes_[i])] for i in range(n_classes)},
    }
    with open(os.path.join(args.out, 'transfer_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---- 5) 写出预测结果（CSV + 区块级 SHP 供 WebGIS）----
    print('[5/5] 写出预测结果...')
    df = pd.DataFrame({
        'block_id': block_ids,
        'true_label': [gz_le.inverse_transform([t])[0] if t >= 0 else 'UNSEEN' for t in trues_gz],
        'pred_label': [gz_le.inverse_transform([p])[0] for p in preds_gz],
        'pred_prob': max_probs,
        'correct': [bool(preds_gz[i] == trues_gz[i]) if trues_gz[i] >= 0 else None
                    for i in range(len(trues_gz))],
    })
    df.to_csv(os.path.join(args.out, 'transfer_predictions.csv'), index=False, encoding='utf-8-sig')

    # 区块级预测图：把预测按 block_id 贴回目标城市建筑（每栋建筑继承其区块预测）
    try:
        import geopandas as gpd
        tgt_cfg = tgt_meta.get('config', {})
        bshp = tgt_cfg.get('data', {}).get('buildings_shp')
        if bshp and os.path.exists(bshp):
            gdf = gpd.read_file(bshp)
            bid_col = tgt_meta.get('building_id_field') or 'block_id'
            # 以 block_id 为键：取每个 block 的预测（取第一个出现的预测即可）
            block_pred = df.dropna(subset=['block_id']).groupby('block_id').first()
            gdf['pred_label'] = gdf[bid_col].map(block_pred['pred_label'])
            gdf['pred_prob'] = gdf[bid_col].map(block_pred['pred_prob'])
            gdf['true_label'] = gdf[bid_col].map(block_pred['true_label'])
            out_shp = os.path.join(args.out, 'transfer_predictions.shp')
            gdf.to_file(out_shp)
            print(f'    区块级预测图写出: {out_shp}')
        else:
            print('    [跳过] 未找到目标建筑 shp，未生成 SHP（仅 CSV）')
    except Exception as e:
        print(f'    [跳过] SHP 写出失败: {e}')

    print(f'\n[完成] 耗时 {time.time() - t0:.1f}s | 指标: {args.out}/transfer_metrics.json')
    print(f'    跨城市精度 = {acc:.4f}  (源城=目标城自测时，该值应接近该城本身的精度)')


if __name__ == '__main__':
    main()
