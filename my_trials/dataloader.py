import pandas as pd
import numpy as np
import torch


# 生成假資料
np.random.seed(0)
data = np.random.rand(30, 6)  # 建立 30 行 6 列的隨機數據

# 將數據轉換為 DataFrame
df = pd.DataFrame(data, columns=['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6'])

# 假設 node_1 影響 node_2 和 node_3
df['node_2'] = df['node_2'] + 0.5 * df['node_1']
df['node_3'] = df['node_3'] + 0.3 * df['node_1']

# 假設 node_4 影響 node_5 和 node_6
df['node_5'] = df['node_5'] + 0.4 * df['node_4']
df['node_6'] = df['node_6'] + 0.6 * df['node_4']

# 印出 DataFrame
print(df)

numpy_array = df.values
transposed_array = numpy_array.T  # Transpose the array
tensor_variable = torch.tensor(transposed_array)

# print("Type of tensor_variable:", type(tensor_variable))
# print("Shape of tensor_variable:", tensor_variable.shape)
# print("Data of tensor_variable:")
# print(tensor_variable)


###############
matrix1 = [
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0]
]

matrix2 = [
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0]
]

# 將矩陣轉換為 PyTorch Tensor
tensor1 = torch.tensor(matrix1)
tensor2 = torch.tensor(matrix2)

# 放入列表中
meta_path_list = [tensor1, tensor2]
print(f"meta_path_list: {meta_path_list}")

