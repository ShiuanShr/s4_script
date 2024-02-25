import torch
import torch.optim as optim
import numpy as np
import dgl
from dgl import DGLGraph
from model import HAN
import torch.nn as nn
import torch

def generate_node_features(num_nodes = 10, features_count = 1, row=17):
    # 初始化一个二维数组，用于存储生成的特征
    node_features = []

    # 生成每个节点的特征
    for _ in range(num_nodes): # 10 nodes
        # 生成节点的特征
        features = []
        for count in range(features_count): # 2 col
            # 为每个特征生成一组数据
            feature_data = np.random.rand(row) # generate list with shape of (row,)
            print(f"feature_data:={feature_data.shape}")
            features.append(feature_data)
        # 将生成的特征加入节点特征列表中
        node_features.append(features)

    return node_features

# node_features = generate_node_features()
# print(f"node_features: {node_features}, size: {len(node_features)}" )
# print("--------")
# print(f"node_features: {node_features}, size: {len(node_features)}")

num_graphs = 3
num_nodes = 10
node_features_count = 17
feature_dim = 1
adj_matrices = [np.random.randint(0, 2, size=(num_nodes, num_nodes)) for _ in range(num_graphs)]
print(f"adj_matrices: {adj_matrices}, size: {len(adj_matrices)}")
# Convert the features to PyTorch tensors

node_features = torch.tensor([generate_node_features() for _ in range(num_graphs)])

# node_features = torch.tensor([np.random.rand(num_nodes, feature_dim) for _ in range(num_graphs)])
# print(f"node_features special3: {node_features}, size: {node_features.shape}")

def train():
    # 参数设置
    in_dim = 1  # 每个节点的特征维度
    hidden_dim = 128
    out_dim = 128
    num_heads = 8 # num_heads  8* out_dim 128= 第一層的1*1024
    dropout = 0.6
    lr = 0.005
    weight_decay = 0.001
    num_epochs = 200
    patience = 100

    # 初始化模型和优化器
    print(1)
    model = HAN(num_metapaths=num_graphs, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_heads=num_heads, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    print(2)
    # 將數據轉換為DGLGraph對象
    # graphs = [DGLGraph(adj) for adj in adj_matrices]
    # graphs = [dgl.graph(adj) for adj in adj_matrices]
    # graphs = [dgl.graph(adj_matrix) for adj_matrix in adj_matrices]
    graphs = [dgl.graph((adj_matrix != 0).nonzero()) for adj_matrix in adj_matrices] #?
    print(graphs)
    print(3)
    # 將數據轉換為PyTorch Tensor
    features = node_features

    # 是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    # 训练循环
    best_loss = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        model.train()
        print(f"model train completed")
        optimizer.zero_grad()
        embeddings = model(graphs, features)
        print(f"embeddings  completed")
        print(f"epoch: {epoch} mean of first node embeddings: {embeddings[0].mean()}") #embeddings.shape = ([10, 128])

        # # 计算损失（此处需要根据您的任务确定损失函数和目标）
        # # 此处的示例只是一个简单的示例，您需要根据实际情况修改
        # loss = custom_loss(embeddings, target)  # target是您希望模型学习的目标图的表示，可能是一张特定图的节点嵌入表示
        # loss.backward()
        # optimizer.step()

        # # 验证损失
        # with torch.no_grad():
        #     model.eval()
        #     val_embeddings = model(graphs, features)
        #     val_loss = criterion(val_embeddings, target)

        # print('Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss.item()))

        # # Early stopping
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping after {epoch} epochs.")
        #         break

    # 保存模型
    torch.save(model.state_dict(), 'han_model.pth')

def custom_loss(predictions, targets, specified_node_idx):
    # 计算目标节点特征预测误差部分（MSE）
    target_loss = torch.mean((predictions[:, specified_node_idx] - targets[:, specified_node_idx])**2)
    
    # 计算其他节点特征变化关系部分
    other_nodes_loss = 0
    for i in range(predictions.shape[1]):
        if i != specified_node_idx:
            # 计算指定节点特征增加0.05时，其他节点特征的预期值
            expected_value = targets[:, i] + 0.05
            # 计算模型预测的其他节点特征与预期值之间的差异
            other_nodes_loss += torch.mean((predictions[:, i] - expected_value)**2)
    
    # 返回总损失，可以根据实际情况调节两部分损失的权重
    total_loss = target_loss + other_nodes_loss
    return total_loss

if __name__ == '__main__':
    train()
