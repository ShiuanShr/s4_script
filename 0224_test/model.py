# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAttentionLayer(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, dropout, alpha):
        super(NodeAttentionLayer, self).__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.weight = nn.Parameter(torch.empty(size=(self.in_feature_dim, self.out_feature_dim)))
        self.attention_coef = nn.Parameter(torch.empty(size=(self.out_feature_dim * 2, 1)))
        nn.init.xavier_uniform_(self.weight, gain=1.387)
        nn.init.xavier_uniform_(self.attention_coef, gain=1.387)

    def forward(self, x, adj):
        Wh = torch.mm(x, self.weight)
        e = self._prepare_attention(Wh)
        infneg_vector = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, infneg_vector)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def _prepare_attention(self, Wh):
        Wh1 = torch.matmul(Wh, self.attention_coef[:self.out_feature_dim, :])
        Wh2 = torch.matmul(Wh, self.attention_coef[self.out_feature_dim:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class SemanticAttentionLayer(nn.Module):
    def __init__(self, in_feature_dim, q_vector):
        super(SemanticAttentionLayer, self).__init__()
        self.weight = nn.Parameter(torch.empty(size=(in_feature_dim, q_vector)))
        self.bias = nn.Parameter(torch.empty(size=(1, q_vector)))
        self.q = nn.Parameter(torch.empty(size=(q_vector, 1)))
        nn.init.xavier_uniform_(self.weight, gain=1.667)
        nn.init.xavier_uniform_(self.bias, gain=1.667)
        nn.init.xavier_uniform_(self.q, gain=1.667)

    def forward(self, z):
        print(f"[SemanticAttentionLayer][forward]")
        Wh = torch.matmul(z, self.weight) + self.bias
        Wh = F.tanh(Wh)
        w = torch.matmul(Wh, self.q)
        w = w.mean(0)
        beta = F.softmax(w, dim=1)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class HAN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout, num_heads, alpha, q_vector):
        super(HAN, self).__init__()
        self.dropout = dropout
        self.q_vector = q_vector
        self.num_heads = num_heads
        self.attentions = [NodeAttentionLayer(feature_dim, hidden_dim, self.dropout, alpha) for _ in range(num_heads)]
        print(f"dropout: {dropout}")
        print(f"hidden_dim: {hidden_dim}; alpha: {alpha}")
        print(f"q_vector: {q_vector}")
        print(f"num_heads: {num_heads}")
        # print(f"attentions: {self.attentions}")
        for i, attention in enumerate(self.attentions):
            print(f"i: {i}: {attention}")
            self.add_module('attention_{}'.format(i), attention)
            print(f"add {i} attention completed!")
        self.semantic_attention = SemanticAttentionLayer(hidden_dim * num_heads, q_vector) # 8 * 8, 128
        print(f"semantic_attention: {self.semantic_attention}")

    def forward(self, x, adj):
        print(f"Start the forward...")
        semantic_embeddings = []
        for meta_path_adj in adj:
            x_dropouted = [F.dropout(feat, self.dropout, training=self.training) for feat in x]  # 在這裡對特徵向量進行 dropout
            Z = torch.cat([attention(x_dropouted, meta_path_adj) for attention in self.attentions], dim=1)
            Z = F.dropout(Z, self.dropout, training=self.training)
            semantic_embeddings.append(Z)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        final_embedding = self.semantic_attention(semantic_embeddings)
        return final_embedding
