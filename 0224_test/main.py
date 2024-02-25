# main.py
import numpy as np
import torch
import torch.optim as optim
from model import HAN

# 模擬數據
# 假設我們有3張圖，每張圖都是10*10的有向圖，用1和0表示是否有邊相連
# 這裡我們只是隨機生成一些數據作為示例
num_graphs = 3
num_nodes = 10
feature_dim = 1
adj_matrices = [np.random.randint(0, 2, size=(num_nodes, num_nodes)) for _ in range(num_graphs)]
features = [np.random.rand(num_nodes, feature_dim) for _ in range(num_graphs)]
# print(adj_matrices)
# # 將數據轉換為PyTorch tensor
adj_tensors = [torch.FloatTensor(adj) for adj in adj_matrices]
feature_tensors =  [torch.FloatTensor(feature) for feature in features]
print(f"adj_tensors: {adj_tensors}")
print(f"feature_tensors: {feature_tensors}")
# # 初始化模型和優化器
model = HAN(feature_dim=feature_dim, hidden_dim=8, dropout=0.6, num_heads=8, alpha=0.2, q_vector=128)
print(f"model built...{model}")
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
print(f"optimizer built...{optimizer}")

# 訓練模型
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    output = model(feature_tensors, adj_tensors) # 
    print(f"output--------: {output}")
    # 在無監督情況下，這裡可以設計一個自定義的損失函數，比如說計算節點特徵的變化情況等
    loss = torch.mean(torch.abs(output))  # 這裡只是一個示例損失函數
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.item()))

# 推理
model.eval()
output = model(feature_tensors, adj_tensors)