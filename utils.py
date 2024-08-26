import numpy as np
import scipy.sparse as sp
import torch
from sklearn import metrics
from scipy.spatial.distance import pdist
from sklearn.metrics import f1_score

def encode_onehot(labels):
    # labels输入所有数据的标签
    classes = set(labels)
     # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # 创建方阵I（仅有对角线为1）
    '''enumerate()函数生成序列，带有索引i和值c。
    这一句将string类型的label变为int类型的label，建立映射关系
    np.identity(len(classes)) 为创建一个classes的单位矩阵
    创建一个字典，索引为 label， 值为独热码向量（就是之前生成的矩阵中的某一行）'''
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # 为所有的标签生成相应的独热码
    # map() 会根据提供的函数对指定序列做映射。
    # 这一句将string类型的label替换为int类型的label
    return labels_onehot


def load_data(path="../data/planetoid-master/tmp/pubmed/PubMed/", dataset="pubmed"):
    """Load citation network dataset (pubmed only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
                                        # 读入数据，参数一是文件地址fname
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # 切片存储feature和label

    """根据引用文件，生成无向图"""    
    # build graph
    # 将每篇文献的编号提取出来并对其赋予节点编号，存到字典中（i：节点编号，j：文献编号）
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 对文献的编号构建字典
    idx_map = {j: i for i, j in enumerate(idx)}
    #读取cite文件
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
   # 生成图的边，（x,y）其中x、y都是为以文章编号为索引得到的值，此外，y中引入x的文献
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # edges是先把edges_unordered打到一维，用节点编号替换文献编号，之后恢复成edges_unordered的矩阵size

    #生成邻接矩阵，生成的矩阵为稀疏矩阵，对应的行和列坐标分别为边的两个点，该步骤之后得到的是一个有向图

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # coo_matrix((data, (i, j)), [shape=(M, N)])
    # 这里有三个参数：
    # data[:] 就是原始矩阵中的数据，例如上面的4,5,7,9；

    # i[:] 就是行的指示符号；例如上面row的第0个元素是0，就代表data中第一个数据在第0行；

    # j[:] 就是列的指示符号；例如上面col的第0个元素是0，就代表data中第一个数据在第0列；

    # build symmetric adjacency matrix
    # 转成对称阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    #进行归一化，对应于论文中的A^=(D~)^0.5 A~ (D~)^0.5,但是本代码实现的是A^=(D~)^-1 A~
    #A^=I+A,sp.eye是单位矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # range() 函数创建一个整数列表，构建训练集、验证集、测试集，
    # 创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    #todense转化为numpy.matrix
    features = torch.FloatTensor(np.array(features.todense()))
    # 之前将labels转换为one-hot后，再转化为数值类型，如[3 1 1 0 1 1 1 1 2 2 2]
    labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # 对传入特征矩阵的每一行分别求和,得到（2708，1）的矩阵
    r_inv = np.power(rowsum, -1).flatten()
    # 求倒数,用flatten打成一维数组返回
    r_inv[np.isinf(r_inv)] = 0.
    # 倒数是无穷大的恢复为0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    # 计算NMI值
    # 使用type_as(tesnor)将张量转换为给定类型的张量。
    preds = output.max(1)[1].type_as(labels)
    predsn = preds.cpu().detach().numpy()
    # A.max(1)：返回A每一行最大值组成的一维数组
    labelsn = labels.cpu().detach().numpy()
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    # return correct / len(labels)
    nmi_data = metrics.normalized_mutual_info_score(predsn,labelsn)
    return nmi_data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def set_truelabel_list(path="../data/eumail/", dataset="eumail", beg_val = 118):
    # 将真实标签和生成标签一样做排列
    # 生成标签注入
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
                                        # 3入content数据，参数一是文件地址fname
    # our data dont have features,just labels
    labels = idx_features_labels[:, -1]
    # print(labels)
    idx = idx_features_labels[:, 0]

    # 切片存储feature和label
    # 读取真实标签并更改排序以与生成标签一致
    idx_true_labels = np.genfromtxt("{}{}.labels".format(path, dataset),
                                        dtype=np.dtype(str))
                                        # 3入content数据，参数一是文件地址fname
    true_labels = idx_true_labels[:, -1]
    true_idx = idx_true_labels[:, 0]
    # true_labels = np.where(true_labels)
    cs_zs_labels = idx_features_labels[:, -1]
    # cs_zs_labels构成：(0-beg_val-1) 生成标签 (beg_val - end) 真实标签
    # cszs_true_labels构成，按照cs_zs生成节点标签顺序排列的，节点真实标签

    for i in range(beg_val-1,true_idx.shape[0]):
        target_id = idx[i]
        for j in range(0,true_idx.shape[0]):
            if target_id == true_idx[j]:
                break
        # if i>=beg_val:
        cs_zs_labels[i] = true_labels[j]
        # cszs_true_labels[i] = true_labels[j]
    
    # cs_zs_labels构成：(0-beg_val-1) 生成标签 (beg_val - end) 真实标签
    cs_zs_labels = encode_onehot(cs_zs_labels)
    cszs_true_labels = idx_features_labels[:, -1]
    for i in range(0,true_idx.shape[0]):
        target_id = idx[i]
        for j in range(0,true_idx.shape[0]):
            if target_id == true_idx[j]:
                break
        cszs_true_labels[i] = true_labels[j]

    cszs_true_labels = encode_onehot(cszs_true_labels)
    return cs_zs_labels,cszs_true_labels

def set_truelabel_list_fake(path="../data/eumail/", dataset="eumail", beg_val = 118):
    # 将真实标签和生成标签一样做排列
    # 生成标签注入
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
                                        # 3入content数据，参数一是文件地址fname
    # our data dont have features,just labels
    labels = idx_features_labels[:, -1]
    idx = idx_features_labels[:, 0]

    # 切片存储feature和label
    # 读取真实标签并更改排序以与生成标签一致
    idx_true_labels = np.genfromtxt("{}{}.labels".format(path, dataset),
                                        dtype=np.dtype(str))
                                        # 3入content数据，参数一是文件地址fname
    true_labels = idx_true_labels[:, -1]
    true_idx = idx_true_labels[:, 0]
    # true_labels = np.where(true_labels)
    cs_zs_labels = idx_features_labels[:, -1]
    cszs_true_labels = idx_features_labels[:, -1]
    # cs_zs_labels构成：(0-beg_val-1) 生成标签 (beg_val - end) 真实标签
    # cszs_true_labels构成，按照cs_zs生成节点标签顺序排列的，节点真实标签

    for i in range(beg_val-1,true_idx.shape[0]):
        target_id = idx[i]
        for j in range(0,true_idx.shape[0]):
            if target_id == true_idx[j]:
                break
        # if i>=beg_val:
        cs_zs_labels[i] = true_labels[j]
        # cszs_true_labels[i] = true_labels[j]


    for i in range(0,true_idx.shape[0]):
        target_id = idx[i]
        for j in range(0,true_idx.shape[0]):
            if target_id == true_idx[j]:
                break
        cszs_true_labels[i] = true_labels[j]

    # cs_zs_labels构成：(0-beg_val-1) 生成标签 (beg_val - end) 真实标签
    cs_zs_labels = encode_onehot(cs_zs_labels)
    cszs_true_labels = encode_onehot(cszs_true_labels)
    # return cs_zs_labels
    return cs_zs_labels,cszs_true_labels

def data_init_node_expression(path="../data/eumail/", dataset="eumail", beg_val =322,beg_test = 500):
    """Load community network dataset (give them node expression)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
                                        # 3入content数据，参数一是文件地址fname
    # 切片存储feature和label
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # our data dont have features,just labels
    # labels = encode_onehot(idx_features_labels[:, -1])
    # labels用函数来生成
    labels,cszs_true_labels = set_truelabel_list(path,dataset,beg_val)

    """根据引用文件，生成无向图"""    
    # build graph
    # 将每篇文献的编号提取出来并对其赋予节点编号，存到字典中（i：节点编号，j：文献编号）
    # 如果有孤立的点呢？3从content文件中读取的，不会有问题
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 对文献的编号构建字典（i从0-n)
    idx_map = {j: i for i, j in enumerate(idx)}
    #读取cite文件
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
   # 生成图的边，（x,y）其中x、y都是为以文章编号为索引得到的值，此外，y中引入x的文献
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # edges是先把edges_unordered打到一维，用节点编号替换文献编号，之后恢复成edges_unordered的矩阵size

    #生成邻接矩阵，生成的矩阵为稀疏矩阵，对应的行和列坐标分别为边的两个点，该步骤之后得到的是一个有向图

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # coo_matrix((data, (i, j)), [shape=(M, N)])
    # 这里有三个参数：
    # data[:] 就是原始矩阵中的数据，例如上面的4,5,7,9；

    # i[:] 就是行的指示符号；例如上面row的第0个元素是0，就代表data中第一个数据在第0行；

    # j[:] 就是列的指示符号；例如上面col的第0个元素是0，就代表data中第一个数据在第0列；

    # build symmetric adjacency matrix
    # 转成对称阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 定义features为邻接矩阵
    features = adj
    features = normalize(features)
    adja = adj.toarray()
    #进行归一化，对应于论文中的A^=(D~)^0.5 A~ (D~)^0.5,但是本代码实现的是A^=(D~)^-1 A~
    #A^=I+A,sp.eye是单位矩阵
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    # range() 函数创建一个整数列表，构建训练集、验证集、测试集，
    # 创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    # 到时候要修改的
    idx_train = range(beg_val -1)
    idx_train = np.fromiter(iter(idx_train),dtype='int')
    idx_val = range(beg_val, beg_test)
    idx_val = np.fromiter(iter(idx_val),dtype='int')
    idx_test = range(beg_test, idx.shape[0])
    idx_test = np.fromiter(iter(idx_test),dtype='int')
    #todense转化为numpy.matrix
    features = np.array(features.todense())
    # 之前将labels转换为one-hot后，再转化为数值类型，如[3 1 1 0 1 1 1 1 2 2 2]
    labels = np.where(labels)[1]
    cszs_true_labels = np.where(cszs_true_labels)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # 转化连接矩阵为dict
    # s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
    # d={}
    # for k, v in s:
    #     d.setdefault(k,[]).append(v)
    adjacency_dict = {}
    adja = adj.toarray()
    for i in range(0,adja.shape[0]):
        adjacency_dict.setdefault(i,[])
        for j in range(0,adja.shape[0]):
            if(adja[i][j]!=0):
                adjacency_dict.setdefault(i,[]).append(j)

    return adj,adjacency_dict, features, labels,cszs_true_labels, idx_train, idx_val, idx_test

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


def loss_full(emb, adj,num_nodes=1005,num_edges=25571):
    # 使用时要num_edges,num_nodes
    num_nonedges= num_nodes**2 - num_nodes - num_edges
    edge_proba = num_edges / (num_nodes**2 - num_nodes)
    eps = -np.log(1 - edge_proba)
    """Compute BerPo loss for all edges & non-edges in a graph."""
    e1, e2 = adj.nonzero()
    edge_dots = torch.sum(emb[e1] * emb[e2])
    # print("--edge_dots: ", edge_dots)
    # print("--eps",eps)
    loss_edges = -torch.sum(torch.log(-torch.expm1(-eps - edge_dots)))
    # print("--loss_edges: ", loss_edges)

    # Correct for overcounting F_u * F_v for edges and nodes with themselves
    # self_dots_sum = torch.sum(emb * emb)
    # correction = self_dots_sum + torch.sum(edge_dots)
    # sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
    # loss_nonedges = torch.sum(emb @ sum_emb) - correction


    loss_nonedges= torch.sum(emb * emb) - torch.sum(emb[e1] * emb[e2])
    # print("loss_nonedges: ", loss_nonedges)
    neg_scale = num_nonedges / num_edges
    return (loss_edges / num_edges + neg_scale * loss_nonedges / num_nonedges) / (1 + neg_scale)
    # return (loss_edges / num_edges + loss_nonedges / num_nonedges) 


def evaluate_score( output, labels, type=0):
    # 估计准确度
    # type = 1 F1相似度， type = 0 Jaccard相似度
    # 使用type_as(tesnor)将张量转换为给定类型的张量。
    preds = output.max(1)[1].type_as(labels)
    predsn = preds.cpu().detach().numpy()
    labelsn = labels.cpu().detach().numpy()
    x = predsn
    y = labelsn
    x_set = set(x)
    x_set = list(x_set)
    y_set = set(y)
    y_set = list(y_set)
    cx_len = len(x_set)
    cy_len = len(y_set)
    evaluate_data = 0
    output_ans = 0
    for i in x_set: 
        # print("i in x_set",i)
        max_score = 0
        # 为每一个output找最接近的labels
        xsele = np.where(x==i,1,0)
        xsele_add = np.where(x==i)
        for j in y_set:
            calans = 0
            ysele = np.where(y==j,1,0)
            ysele_add = np.where(y==j)
            if type:
                calans = f1_score(xsele, ysele, average='binary')
                # print("calans",calans)
            else:
                # jaccard
                #相同的维度的个数占所有维度的比例
                AL = set(xsele_add[0])
                BL = set(ysele_add[0])
                calans = len(AL&BL)/len(AL|BL) 
            if calans >= max_score:
                max_score = calans
        # print("score of i",max_score)
        output_ans = output_ans +max_score
    evaluate_data = output_ans / (2 *cx_len)
    output_ans = 0
    for i in y_set: 
        max_score = 0
        # 为每一个output找最接近的labels
        ysele = np.where(y==i,1,0)
        ysele_add = np.where(y==i)
        for j in x_set:
            # print("round2:searching for y at x",j,i)
            calans = 0
            xsele = np.where(x==j,1,0)
            xsele_add = np.where(x==j)
            if type:
                calans = f1_score(xsele, ysele, average='binary')
            else:
                # jaccard
                #相同的维度的个数占所有维度的比例
                AL = set(xsele_add[0])
                BL = set(ysele_add[0])
                calans = len(AL&BL)/len(AL|BL)
            if calans>max_score:
                max_score = calans
        output_ans = output_ans +max_score
    evaluate_data = evaluate_data + output_ans / (2 *cy_len)
    return evaluate_data