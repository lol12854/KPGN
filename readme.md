## KPGN
KPGN的相关数据放在data/xxx/下，包括xx.cites、xx.labels和xx.content，分别表示边信息、真实标签和伪标签信息，伪标签信息通过gen_kplex得到。
该文件夹中的main_xx.py都是为了方便不同数据集调参稍作改变得到的，
main_enkg 适配了ENKG中新增的数据集
## gen_kplex
该文件夹中的KSKP_read_xx.py都是为了适配不同数据集稍作改变得到的,标签
KSKP_read_mat 用来读取根目录下mat文件表示的数据集，在/mat_data文件夹中可以找到
KSKP_read_pc 用来读取pubmed，citeseer，cora数据集，在/data文件夹中有相关数据
KSKP_read_pkl 用来读取用pickle保存的networkx数据集并且得到k-plex结果,在/pkl_data文件夹中可以找到
KSKP_read_mat 用来读取gms文件表示的数据集，如‘adddata21\\dolphins.gml’
KSKP_read_mat 用来读取csv文件表示的数据集，如‘adddata21\\dp815.csv’
