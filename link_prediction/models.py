import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransE(nn.Module):
    """TransE 模型：得分 = -||h + r - t||"""
    def __init__(self, num_entities, num_relations, embedding_dim, gamma=12.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gamma = gamma

        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # 初始化
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # 对嵌入进行归一化（TransE 常用技巧）
        self._normalize_embeddings()

    def _normalize_embeddings(self):
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)
            self.relation_emb.weight.data = F.normalize(self.relation_emb.weight.data, p=2, dim=1)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        score = -torch.norm(h + r - t, p=2, dim=1)
        return score

    def score_triplet(self, h_idx, r_idx, t_idx):
        return self.forward(h_idx, r_idx, t_idx)


class DistMult(nn.Module):
    """DistMult 模型：得分 = sum(h * r * t)"""
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        score = torch.sum(h * r * t, dim=1)
        return score

    def score_triplet(self, h_idx, r_idx, t_idx):
        return self.forward(h_idx, r_idx, t_idx)


class ComplEx(nn.Module):
    """ComplEx 模型：实体和关系为复数嵌入，得分 = Re(sum(h * r * conj(t)))"""
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim  # 实际嵌入维度为 2*embedding_dim

        self.entity_emb = nn.Embedding(num_entities, 2 * embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, 2 * embedding_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)

        h_re, h_im = torch.chunk(h, 2, dim=-1)
        r_re, r_im = torch.chunk(r, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)

        # ComplEx 得分公式
        score = (h_re * r_re * t_re) + \
                (h_re * r_im * t_im) + \
                (h_im * r_re * t_im) - \
                (h_im * r_im * t_re)
        return score.sum(dim=-1)

    def score_triplet(self, h_idx, r_idx, t_idx):
        return self.forward(h_idx, r_idx, t_idx)


class RotatE(nn.Module):
    """RotatE 模型：复数空间旋转，得分 = -||h ◦ r - t||"""
    def __init__(self, num_entities, num_relations, embedding_dim, gamma=12.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim  # 实体嵌入维度为 2*embedding_dim
        self.gamma = gamma

        self.entity_emb = nn.Embedding(num_entities, 2 * embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)  # 关系只存相位（角度）

        nn.init.xavier_uniform_(self.entity_emb.weight)
        # 关系嵌入初始化为0附近，再经过 cos/sin 映射
        nn.init.uniform_(self.relation_emb.weight, a=-np.pi, b=np.pi)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r_angle = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)

        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)

        # 复数旋转：h * r
        r_re = torch.cos(r_angle)
        r_im = torch.sin(r_angle)

        h_rot_re = h_re * r_re - h_im * r_im
        h_rot_im = h_re * r_im + h_im * r_re

        # 计算距离
        dist = torch.sqrt((h_rot_re - t_re)**2 + (h_rot_im - t_im)**2 + 1e-12)
        score = -dist.sum(dim=-1)
        return score

    def score_triplet(self, h_idx, r_idx, t_idx):
        return self.forward(h_idx, r_idx, t_idx)


class ConvE(nn.Module):
    """ConvE baseline: 2D convolution over reshaped head and relation embeddings."""
    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim,
        emb_shape=(16, 8),
        num_filters=32,
        kernel_size=3,
        input_dropout=0.2,
        feature_map_dropout=0.2,
        hidden_dropout=0.3,
    ):
        super().__init__()
        if emb_shape[0] * emb_shape[1] != embedding_dim:
            raise ValueError("emb_shape[0] * emb_shape[1] must equal embedding_dim.")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.emb_shape = emb_shape
        self.num_filters = num_filters

        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.tail_bias = nn.Embedding(num_entities, 1)

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        conv_height = 2 * emb_shape[0]
        conv_width = emb_shape[1]
        self.fc = nn.Linear(num_filters * conv_height * conv_width, embedding_dim)

        self.input_dropout = nn.Dropout(input_dropout)
        self.feature_map_dropout = nn.Dropout2d(feature_map_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.zeros_(self.tail_bias.weight)

    def encode_pair(self, h_idx, r_idx):
        h = self.entity_emb(h_idx).view(-1, 1, self.emb_shape[0], self.emb_shape[1])
        r = self.relation_emb(r_idx).view(-1, 1, self.emb_shape[0], self.emb_shape[1])

        x = torch.cat([h, r], dim=2)
        x = self.input_dropout(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.hidden_dropout(x)
        return x

    def forward(self, h_idx, r_idx, t_idx):
        x = self.encode_pair(h_idx, r_idx)
        return self.score_encoded_triplet(x, t_idx)

    def score_triplet(self, h_idx, r_idx, t_idx):
        return self.forward(h_idx, r_idx, t_idx)

    def score_encoded_triplet(self, pair_emb, t_idx):
        t = self.entity_emb(t_idx)
        return torch.sum(pair_emb * t, dim=1) + self.tail_bias(t_idx).squeeze(-1)

    def score_encoded_tails(self, pair_emb, tail_idx):
        if tail_idx.dim() == 1:
            tails = self.entity_emb(tail_idx)
            scores = pair_emb @ tails.transpose(0, 1)
            return scores + self.tail_bias(tail_idx).view(1, -1)

        if tail_idx.dim() != 2:
            raise ValueError("tail_idx must be a 1D or 2D tensor.")

        tails = self.entity_emb(tail_idx)
        scores = torch.sum(pair_emb.unsqueeze(1) * tails, dim=-1)
        return scores + self.tail_bias(tail_idx).squeeze(-1)

    def score_tails(self, h_idx, r_idx, tail_idx):
        """
        Score each (h, r) pair against many candidate tails without repeating
        the expensive ConvE convolution for every tail.

        tail_idx can be [num_candidates] for a shared candidate set or
        [batch_size, num_candidates] for per-row candidates.
        """
        x = self.encode_pair(h_idx, r_idx)
        return self.score_encoded_tails(x, tail_idx)
