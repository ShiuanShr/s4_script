import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dgl
from dgl.nn import GATConv
import matplotlib.pyplot as plt

# 生成随机图数据
num_nodes = 10
num_edges = 20
num_features = 17
num_heads = 4
hidden_layer = 60
# 创建随机图
g = dgl.rand_graph(num_nodes, num_edges)
g = dgl.add_self_loop(g)

# 随机初始化节点特征
features_per_node = 1
node_features = torch.randn(num_nodes, features_per_node, num_features)  # 每个节点有2个特征，每个特征有17个数据点
print(f"node_features: {node_features}shape: {node_features.shape}") # [10, 2, 17]
g.ndata['feat'] = node_features

# 定义 GAT 模型
class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(GATModel, self).__init__()
        print(f"""in_dim: {in_dim}, hidden_dim: {hidden_dim}, num_heads: {num_heads}""")
        print(11)
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads) # 17 64 4 
        print("cpv1 done1")
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        # hidden 64 * num_heads 4 = 256
        print("cpv2 done1")

    def forward(self, g, features):
        print(f"forward")
        x = features
        x = self.conv1(g, x).relu()
        print(f"cov1: {x.shape}") # ([10, 2, 4, 64]
        #  RuntimeError: mat1 and mat2 shapes cannot be multiplied ((num_nodes*features_per_node )x (hidden) and (hidden*num_heads)(hidden*num_heads))
        # 也就是說這邊要乘上 num_heads
        x = self.conv2(g, x).relu()
        print(f"cov2")
        return x

# 初始化 GAT 模型
model = GATModel(num_features, hidden_layer, num_heads)

print(f"model built: {model}")
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
losses = []
for epoch in range(num_epochs):
    # 前向传播
    print(f"epoch: {epoch}")
    logits = model(g, g.ndata['feat'])
    
    # 计算损失
    loss = torch.mean(logits)
    losses.append(loss.item())
    
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 输出当前训练进度
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 可视化损失曲线
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
