# 創建 HAN 模型
from models import HAN
import torch
import torch.nn as nn
import argparse
import numpy as np
from dataloader import tensor_variable, meta_path_list
import torch.optim as optim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=88,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='Number of hidden dimension.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads for node-level attention.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu.')
    parser.add_argument('--q_vector', type=int, default=128,
                        help='The dimension for the semantic attention embedding.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Number of epochs with no improvement to wait before stopping')
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='The dataset to use for the model.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    features = tensor_variable 
    model = HAN(feature_dim=features.shape[1], #30
                hidden_dim= args.hidden_dim, # 8
                num_classes=None, 
                dropout= args.dropout, # 0.6
                num_heads= args.num_heads, # 8
                alpha= args.alpha, # 0.2
                q_vector= args.q_vector) # 128
        
    # 定義損失函數
    criterion = nn.CrossEntropyLoss()

    # 定義優化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 訓練模型
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        # 將特徵和相應的邊列表示為模型的輸入
        output = model(tensor_variable, meta_path_list)
        # 在這裡，你可以將labels作為參數傳遞到output中，如果你有的話
        # loss = criterion(output, labels)
        # 反向傳播和優化
        # loss.backward()
        optimizer.step()

        # 每隔一段時間打印一次訓練狀態
        if epoch % print_every == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, loss.item()))

    print('Finished Training')


# # 定義損失函數
# criterion = nn.CrossEntropyLoss()
# num_epochs = 10
# learning_rate = 0.001
# # 定義優化器
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(features, meta_path_list)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     # 在每個 epoch 結束時進行驗證
#     with torch.no_grad():
#         model.eval()
#         val_outputs = model(val_features, val_meta_path_list)
#         val_loss = criterion(val_outputs, val_labels)
#         # 可以在此處打印驗證集的準確率等信息
