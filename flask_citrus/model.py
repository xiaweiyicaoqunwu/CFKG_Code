import torch
import torch.nn as nn


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

