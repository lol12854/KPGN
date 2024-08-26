import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from net import GraphSage
from sampling import multihop_sampling
from utils import data_init_node_expression,accuracy,evaluate_score,loss_full

from collections import namedtuple

import numpy as np
from scipy.spatial.distance import pdist
def similary_sampling(adj_dict,nodes_num,getnei_num=20):
    nei_adj_dict = {}
    for i in range(0,nodes_num):
        nei_adj_dict.setdefault(i,[])
        nei_i = adj_dict[i]
        node_i_degree = len(nei_i)
        if node_i_degree<= getnei_num:
            for k in nei_i:
                nei_adj_dict.setdefault(i,[]).append(k)
            
        else:
            i_nei_list = []
            for j in adj_dict[i]:
                j_sim_list = []
                j_sim_list.append(j)
                nei_j = adj_dict[j]
                node_j_degree = len(nei_j)
                # 寻找交集
                similen_ij = len(list(set(nei_i).intersection(nei_j)))
                similary_ij = (node_i_degree-1)*(node_j_degree-1)/((node_j_degree+node_i_degree-2-similen_ij)**2)
                j_sim_list.append(similary_ij)
                i_nei_list.append(j_sim_list)
                # return
            i_nei_list.sort(reverse=True,key=takeSecond)
            for k in range(0,getnei_num):
                sele_neik = i_nei_list[k][0]
                nei_adj_dict.setdefault(i,[]).append(sele_neik)
    return nei_adj_dict    
            

def takeSecond(elem):
    return elem[1]


def jaccard(p,q):
    a = 0
    b = 0
    v = 0
    c = [a for i in p if v in b]
    return float(len(c))/(len(a)+len(b)-len(b))

if __name__ == "__main__":
    x = [1,1,2,0,0,4,5,6,6]
    print(set(x))
    # adj4loss,adjacency_dict, features, labels,data_labels, idx_train, idx_val, idx_test = data_init_node_expression(path="../data/eumail/", dataset="eumail",beg_val=487,beg_test = 588)

    # sele_adj_dict = similary_sampling(adj_dict = adjacency_dict,nodes_num = labels.shape[0],getnei_num=20)
    # print(sele_adj_dict)



    # x=np.random.random(8)>0.5
    # y=np.random.random(8)>0.5
    # x = [1,1,0,0,1,1,1]
    # y = [1,1,0,0,1,1,0]

    # x=np.asarray(x,np.int32)
    # y=np.asarray(y,np.int32)

    # #方法一：根据公式求解
    # up=np.double(np.bitwise_and((x != y),np.bitwise_or(x != 0, y != 0)).sum())
    # down=np.double(np.bitwise_or(x != 0, y != 0).sum())
    # d1=(up/down)
    # print("d1",d1)
            

    # #方法二：根据scipy库求解
    # X=np.vstack([x,y])
    # d2=pdist(X,'jaccard')
    # print("d2",d2)

