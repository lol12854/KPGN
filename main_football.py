import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from net import GraphSage
from sampling import multihop_sampling
from utils import data_init_node_expression,accuracy,evaluate_score,similary_sampling

from collections import namedtuple
from torch.optim import lr_scheduler


adj4loss,adjacency_dict, features, labels,data_labels, idx_train, idx_val, idx_test = data_init_node_expression(path="data/football/", dataset="football",beg_val=95,beg_test = 103)

sele_adj_dict = similary_sampling(adj_dict = adjacency_dict,nodes_num = labels.shape[0],getnei_num=30)
# 利用相似度做采样
# sele_adj_dict = adjacency_dict

nclass=labels.max().item() + 1
ALL_NODES = 115 #总节点数
ALL_EDGES = 613 #总边数
INPUT_DIM = ALL_NODES    # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [256, 128, nclass]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [20,10,5]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
EPOCHS = 100

LEARNING_RATE = 0.001     # 学习率
DEVICE = "cuda"
# "cuda"

"""返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""


x = features / features.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1 需要？
all_index = range(ALL_NODES)
all_index = np.fromiter(iter(all_index),dtype='int')

train_index = idx_train
train_label = labels
test_index = idx_test
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
print(model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def loss_full(emb, adj,num_nodes=1005,num_edges=25571):
    # 使用时要num_edges,num_nodes
    num_nonedges= num_nodes**2 - num_nodes - num_edges
    edge_proba = num_edges / (num_nodes**2 - num_nodes)
    eps = -np.log(1 - edge_proba)
    """Compute BerPo loss for all edges & non-edges in a graph."""
    e1, e2 = adj.nonzero()
    edge_dots = torch.sum(emb[e1] * emb[e2])
    loss_edges = -torch.sum(torch.log(-torch.expm1(-eps - edge_dots)))
    loss_nonedges= torch.sum(emb * emb) - torch.sum(emb[e1] * emb[e2])
    neg_scale = num_nonedges / num_edges
    return (loss_edges / num_edges + 0.05*neg_scale * loss_nonedges / num_nonedges) / (1 + neg_scale) 

def train():
    model.train()
    for e in range(EPOCHS):
        print(e)
        batch_src_index = train_index
        # all_index
        batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
        batch_sampling_result = multihop_sampling(all_index, NUM_NEIGHBORS_LIST, sele_adj_dict)
        batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
        batch_all_logits = model(batch_sampling_x)
        batch_train_logits = batch_all_logits[idx_train]
        loss = criterion(batch_train_logits, batch_src_label)
        loss_martix = loss_full(batch_all_logits,adj4loss,ALL_NODES,ALL_EDGES)
        optimizer.step()  # 使用优化方法进行梯度更新
        # scheduler.step() # 学习率调整
        print("Epoch {:03d}  Loss: {:.4f}".format(e,  loss.item()))

        test()


def test():
    model.eval()
    # 以所有的数据用作test
    with torch.no_grad():
        test_sampling_result = multihop_sampling(all_index, NUM_NEIGHBORS_LIST, sele_adj_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data_labels[all_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        nmi_all = accuracy(test_logits, test_label)
        f1_similary_all = evaluate_score(test_logits, test_label,type=1)
        jacc_similary_all = evaluate_score(test_logits, test_label,type=0)
        # loss_martix = loss_full(test_logits,adj4loss,ALL_NODES,ALL_EDGES)
        print("Test Accuracy: ", accuarcy)
        print("NMI Score: ", nmi_all)
        print("f1_similary_all: ", f1_similary_all)
        print("jacc_similary_all: ", jacc_similary_all)
        

if __name__ == '__main__':
    train()


