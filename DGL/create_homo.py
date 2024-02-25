import dgl
import torch

src_node_id = torch.tensor([0,0,0,1,1,2,2,3])
dist_node_id = torch.tensor([1,2,4,2,3,3,4,4])
graph = dgl.graph((src_node_id,dist_node_id))
print(graph)