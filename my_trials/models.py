import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import NodeAttentionLayer, SemanticAttentionLayer

class HAN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout, num_heads, alpha, q_vector):
        super(HAN, self).__init__()
        self.dropout = dropout
        self.q_vector = q_vector
        self.num_heads = num_heads
        
        # 定義注意力層
        self.attentions = nn.ModuleList([
            NodeAttentionLayer(feature_dim, hidden_dim, dropout, alpha) for _ in range(num_heads)
        ])
        
        # 定義語義注意力層
        self.semantic_attention = SemanticAttentionLayer(hidden_dim * num_heads, q_vector)
        
        # 定義全連接層
        self.out_layer = None
        if num_classes is not None:
            self.out_layer = nn.Linear(hidden_dim * num_heads, num_classes)


    def forward(self, x, adjacency_matrices):
        # x: (batch_size, num_nodes, feature_dim)
        # adjacency_matrices: list of adjacency matrices for different graphs
        
        # 生成每個節點的嵌入
        node_embeddings = []
        for adj_matrix in adjacency_matrices:
            # 對每個圖的每個注意力頭計算嵌入
            graph_embeddings = []
            for attention in self.attentions:
                graph_embeddings.append(attention(x, adj_matrix))
            # 將注意力頭的嵌入拼接在一起
            graph_embeddings = torch.cat(graph_embeddings, dim=1)
            node_embeddings.append(graph_embeddings)
        
        # 將不同圖的嵌入整合，生成最終的嵌入
        semantic_embeddings = torch.stack(node_embeddings, dim=1)
        final_embedding = self.semantic_attention(semantic_embeddings)

        # 通過全連接層進行分類
        output = self.out_layer(final_embedding)

        return output
