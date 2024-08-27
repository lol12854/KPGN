#%%
import networkx as nx
import pickle as pkl
import scipy.io as scio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
import argparse
from glob import glob
import scipy.sparse as sp
import copy
from tqdm import tqdm
#from networkx.generators.community import LFR_benchmark_graph
# from scipy.sparse.linalg.eigen.arpack import eigsh
# 用来读取用pickle保存的networkx数据集并且得到k-plex结果

def graph_reader(input_path):
    """
    从边csv生成networkx 图
    Function to read graph from input path.
    :param input_path: Graph read into memory.
    :return graph: Networkx graph.
    """
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

# def graph_build(ori_g_path,knn_g_path):
#     """
#     从边和knn的txt生成networkx 图
#     Function to read graph from input path.
#     :param input_path: Graph read into memory.
#     :return graph: Networkx graph.
#     """
#     edges_knn = pd.read_table(knn_g_path,sep='\s+')
#     edges_knn = edges_knn.values.tolist()
#     edges_ori = pd.read_table(ori_g_path,sep='\s+')
#     edges_ori = edges_ori.values.tolist()
#     edges_ori.extend(edges_knn)
#     graph = nx.from_edgelist(edges_ori) # 生成图
#     return graph


def runMain(k, lamda,nameofdata,loadG):
    #k = 1
    #lamda = 6
    #构建网络
    # files = '*.mat'
    files = nameofdata #pubmed,cora
    # -----------
    G = copy.deepcopy(loadG)
    G_initial = copy.deepcopy(loadG)
    G_core = copy.deepcopy(loadG)

    dataFile = nameofdata
    print("The Procecing Dataset is ",dataFile," File's storage path is:")
    OutFilename = "data\\planetoid-master\\tmp\\pubmed\\PubMed\\" + dataFile + 'k' + str(k) + 'l' + str(lamda) + ".content"
    print(OutFilename)
    common = []
    max_clique_size = G_initial.number_of_nodes()
    # 最大的clique可以有整张图一样大
    G_initial.add_nodes_from(list(G_initial.nodes()), community='None')#初始化G_initial所有点的属性为None
    
    # 使用kCore来删除度不够的节点(m-k是成为k-plex最小的节点度要求)，并放到GACore里边
    def kCore(m, k, GNet, Gn):
        flag = 0
        b = list(Gn.nodes())
        for node in b:
            if nx.degree(Gn, node) < (m-k):
                Gn.remove_node(node)
                flag = 1
        if flag == 0:
            return Gn
        else:
            #print(m, k, GNet.number_of_nodes(), Gn.number_of_nodes())
            return kCore(m, k, GNet, Gn)
    GACore = kCore(lamda, k, G_initial, G)#G变化，G_initial不变
    # GACOre是删掉度不够的节点得到的，之后在GACore上做查找clique的操作
    start = time.time()
    #利用k-plex找基础网络
    #cliques=[c for c in nx.find_cliques(G)]
    
    for a in tqdm(range(0, GACore.number_of_nodes())):
        if (max_clique_size >= lamda):
            cliques = [c for c in nx.find_cliques(GACore)]
            if (cliques == []):
                clique_num = a-1 #只有大于阈值的团
                break
            #print cliques
            num_cliques = len(cliques)
            clique_sizes = [len(c) for c in cliques]
            max_clique_size = max(clique_sizes)
            max_cliques = [c for c in cliques if len(c) == max_clique_size]
            max_clique = max_cliques[0]#choose the first one in cliques as the max_clique
            if(max_clique_size <= lamda-1): # 判断选择出来的clique的价值，是否符合最小限制
                clique_num = a
                break
            G_core.remove_nodes_from(max_clique)#Delete the nodes of max_clique
            GACore.remove_nodes_from(max_clique)
            max_size = max_clique_size
            for no in GACore.nodes():
                nei = G_initial.neighbors(no)
                nu_maxcli = [l for l in max_clique if l in nei] #节点no的nei和max_clique的交集
                num_maxcli = len(nu_maxcli)
                if ((max_size-num_maxcli) <= (k-1)):
                    max_size = max_size + 1
                    max_clique.append(no)
            G_core.remove_nodes_from(max_clique)
            GACore.remove_nodes_from(max_clique)
            common.append(max_clique)
            G_initial.add_nodes_from(max_clique, community=a)
    clique_num = len(common) - 1
    print("Nodes number in G_core",G_core.number_of_nodes())
    print("in GACore(except degree not fit)",GACore.number_of_nodes())
    print("common_num")
    print(len(common))
    ii = 1
    f = open(OutFilename, 'wt')
    for co in common:
        for no in co:
            f.write(str(no)+" "+str(ii)+'\n')
        ii = ii + 1
    # 输出没有统计的部分(可选是否输出)
    # Still_in_node = G_core.nodes()
    # for no in Still_in_node:
    #     f.write(str(no)+" "+'0'+'\n')
    f.close
    end = time.time()
    print("Ustage of the time:",end-start)
    #------------


# Parameters
# m(lamda)(bound) 代表社团中节点数量的下限
# k 代表k-plex相对clique的松弛程度
kk = [2,3,4,5,6,7,8,9]
mm = [5]
# loadG = nx.read_edgelist("ENKG_ZDB_2023\\2023Code_DB\\2101KP_GS\\gs\\data\\citeseer\\citeseer.txt")
loadG = nx.read_edgelist("data\\planetoid-master\\tmp\\pubmed\\PubMed\\pubmed.cites")

loadG = nx.convert_node_labels_to_integers(loadG)
print('\n')
print(loadG)
print('\n')
#pos = nx.spring_layout(loadG)

# input_path_ori = 'refdata/wa_a.txt'
# input_path_knn = 'refdata/wa_graph.txt'
# loadG = graph_build(input_path_ori,input_path_knn)
# path = 'adddata21\DD6_edge.csv'
# loadG = graph_reader(path)
nameofdata = 'pubmed'
for m in mm:
    for k in kk:
        print("k= ",k,"||","Lamda(M)= ",m)
        runMain(k, m, nameofdata, loadG)


# test
# 默认networkx输入的是无权图，所以两个不同表达的边会被合并
edges=[(0,1),(1,0)]
graph = nx.from_edgelist(edges)
print([e for e in graph.edges])



"""
读取txt文件
该文本中的分割符既有空格又有制表符（‘/t’），sep参数用‘/s+’，可以匹配任何空格。
"""
# input_path_ori = 'refdata/wa_a.txt'
# input_path_knn = 'refdata/wa_graph.txt'
# edges_knn = pd.read_table(input_path_knn,sep='\s+')
# edges_knn = edges_knn.values.tolist()
# edges_ori = pd.read_table(input_path_ori,sep='\s+')
# edges_ori = edges_ori.values.tolist()
# # edges = [[0, 110], [0, 57]]
# # edges_ori = [[1, 121], [1, 175]]
# # edges.extend(edges_ori)
# print(len(edges_ori),len(edges_knn))
# edges_ori.extend(edges_knn)
# print(len(edges_ori))
# graph = nx.from_edgelist(edges_ori) # 生成图

