import torch
import torch.optim as optim
import pickle
import numpy as np
import os
import argparse
import torch.nn.functional as F 
from models import TransE, DistMult, ComplEx, RotatE, ConvE


# --- 配置 ---  
DATA_DIR = "./data"
EMBEDDING_DIM = 128
BATCH_SIZE = 1024
EPOCHS = 200
LR = 0.005
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 负采样参数
NEG_K = 256
ADVERSARIAL_TEMP = 1.0


def load_dataset():
    with open(os.path.join(DATA_DIR, 'dataset.pkl'), 'rb') as f:
        return pickle.load(f)


def train(model_name):
    data = load_dataset()
    num_entities = data['num_nodes']
    num_relations = data['num_rels']
    train_triplets = data['train']

    # 初始化模型
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

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"开始训练 {model_name} ... 设备: {DEVICE}")
    print(f"实体数: {num_entities}, 关系数: {num_relations}, 训练边数: {len(train_triplets)}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        idxs = np.random.permutation(len(train_triplets))

        for i in range(0, len(train_triplets), BATCH_SIZE):
            batch_idx = idxs[i:i+BATCH_SIZE]
            batch = train_triplets[batch_idx]

            h = torch.tensor(batch[:, 0], dtype=torch.long).to(DEVICE)
            r = torch.tensor(batch[:, 1], dtype=torch.long).to(DEVICE)
            t = torch.tensor(batch[:, 2], dtype=torch.long).to(DEVICE)

            # 正样本得分
            pos_scores = None  # Set in the model-specific scoring branch below.
            current_bs = len(h)

            # 负采样
            neg_t = torch.randint(0, num_entities, (current_bs, NEG_K), device=DEVICE)

            if hasattr(model, 'score_tails'):
                pair_emb = model.encode_pair(h, r)
                pos_scores = model.score_encoded_triplet(pair_emb, t)
                neg_scores = model.score_encoded_tails(pair_emb, neg_t)
            else:
                h_expand = h.unsqueeze(1).expand(-1, NEG_K).reshape(-1)
                r_expand = r.unsqueeze(1).expand(-1, NEG_K).reshape(-1)
                neg_t_flat = neg_t.reshape(-1)
                neg_scores = model(h_expand, r_expand, neg_t_flat).view(current_bs, NEG_K)

            # 自对抗损失
            pos_loss = -F.logsigmoid(pos_scores).mean()
            neg_weights = F.softmax(neg_scores * ADVERSARIAL_TEMP, dim=1).detach()
            neg_loss_individual = -F.logsigmoid(-neg_scores)
            neg_loss = (neg_weights * neg_loss_individual).sum(dim=1).mean()

            loss = (pos_loss + neg_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 对 TransE / RotatE 进行嵌入归一化（如有需要）
            if model_name in ['TransE', 'RotatE']:
                if hasattr(model, '_normalize_embeddings'):
                    model._normalize_embeddings()

        scheduler.step(total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | LR: {current_lr:.6f}")

    # 保存模型权重
    save_path = os.path.join(DATA_DIR, f'{model_name}_weights.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE'])
    args = parser.parse_args()
    train(args.model)

# python train_baseline.py --model DistMult
