import torch
import pickle
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from baseline_models import TransE, DistMult, ComplEx, RotatE, ConvE


DATA_DIR = "./data"
EMBEDDING_DIM = 128
BATCH_SIZE = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_resources(model_name):
    print(f"加载 {model_name} 模型和数据...")
    with open(os.path.join(DATA_DIR, 'dataset.pkl'), 'rb') as f:
        data = pickle.load(f)

    num_entities = data['num_nodes']
    num_relations = data['num_rels']

    if model_name == 'TransE':
        model = TransE(num_entities, num_relations, EMBEDDING_DIM).to(DEVICE)
    elif model_name == 'DistMult':
        model = DistMult(num_entities, num_relations, EMBEDDING_DIM).to(DEVICE)
    elif model_name == 'ComplEx':
        model = ComplEx(num_entities, num_relations, EMBEDDING_DIM).to(DEVICE)
    elif model_name == 'RotatE':
        model = RotatE(num_entities, num_relations, EMBEDDING_DIM).to(DEVICE)
    elif model_name == 'ConvE':
        model = ConvE(num_entities, num_relations, EMBEDDING_DIM).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    weight_path = os.path.join(DATA_DIR, f'{model_name}_weights.pth')
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    return model, data


def build_filter_dict(data):
    all_triplets = np.concatenate([data['train'], data['valid'], data['test']], axis=0)
    filter_dict = {}
    for h, r, t in all_triplets:
        key = (h, r)
        # 修正：字典方法为 setdefault，而非 setdefault
        filter_dict.setdefault(key, set()).add(t)
    return filter_dict


def evaluate(model, test_triplets, num_entities, filter_dict):
    print(f"开始评估 {len(test_triplets)} 条测试数据...")
    hits1 = hits3 = hits10 = mrr = 0
    total = len(test_triplets)

    test_loader = DataLoader(test_triplets, batch_size=BATCH_SIZE)

    for batch in test_loader:
        h_batch = batch[:, 0].to(DEVICE)
        r_batch = batch[:, 1].to(DEVICE)
        t_batch = batch[:, 2].to(DEVICE)
        bs = len(h_batch)

        # 构造候选尾实体索引（正确做法：传入索引，而非嵌入向量）
        with torch.no_grad():
            t_all_idx = torch.arange(num_entities, device=DEVICE)
            if hasattr(model, 'score_tails'):
                all_scores = model.score_tails(h_batch, r_batch, t_all_idx)
            else:
                h_expand = h_batch.unsqueeze(1).expand(-1, num_entities).reshape(-1)
                r_expand = r_batch.unsqueeze(1).expand(-1, num_entities).reshape(-1)
                t_all_flat = t_all_idx.unsqueeze(0).expand(bs, -1).reshape(-1)
                all_scores = model.score_triplet(h_expand, r_expand, t_all_flat).view(bs, num_entities)

        all_scores = all_scores.cpu().numpy()
        h_cpu = h_batch.cpu().numpy()
        r_cpu = r_batch.cpu().numpy()
        t_cpu = t_batch.cpu().numpy()

        # 过滤已知正确答案
        for i in range(bs):
            h, r, t = h_cpu[i], r_cpu[i], t_cpu[i]
            true_tails = filter_dict.get((h, r), set())
            filter_idx = list(true_tails - {t})
            if filter_idx:
                all_scores[i, filter_idx] = -np.inf

        # 计算排名
        sorted_indices = np.argsort(-all_scores, axis=1)
        for i in range(bs):
            target = t_cpu[i]
            rank = np.where(sorted_indices[i] == target)[0][0] + 1
            mrr += 1.0 / rank
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 10: hits10 += 1

    print("\n" + "=" * 30)
    print(f"{model.__class__.__name__} 评估结果 (Filtered)")
    print("=" * 30)
    print(f"MRR     : {mrr / total:.4f}")
    print(f"Hits@1  : {hits1 / total:.4f}")
    print(f"Hits@3  : {hits3 / total:.4f}")
    print(f"Hits@10 : {hits10 / total:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE'])
    args = parser.parse_args()

    model, data = load_resources(args.model)
    filter_dict = build_filter_dict(data)
    evaluate(model, data['test'], data['num_nodes'], filter_dict)

# python evaluate_baseline.py --model DistMult
