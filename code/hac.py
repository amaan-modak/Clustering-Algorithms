#Python version Python3 ---Might have some errors if running in python2
import numpy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA as mypca
import matplotlib.pyplot as plt

class GDataset(object):
    def __init__(self, gene_id, ground_truth, gene_att):
        self.gene_id = gene_id
        self.ground_truth = ground_truth
        self.gene_att = gene_att

def read(filename):
    global gene_num
    global att_num

    gene_num = 0
    att_num = 0
    
    with open (filename,'r') as file:
        l = next(file)
        att_num = len(l.split('\t'))    
   
        for line in file:
            gene_num+=1
            
    gene_num = gene_num+1 
    att_num = att_num-2 #first two attributes are gene id and cluster id

    id_list = []
    gt_list = []
    att_matrix = numpy.zeros((gene_num,att_num))

    line_num = 0
    with open(filename) as f:
            for line in f:
                entry = line.split('\t')
                
                id_list.append(float(entry[0]))
                gt_list.append(float(entry[1]))
                for i in range (2, att_num+2):
                    att_matrix[line_num][i-2] = float(entry[i])
                line_num += 1

    mydata = GDataset(id_list, gt_list, att_matrix)
    return mydata

def get_min_pos(c_dist,n):
    cmin = c_dist[0][1]
    pos = [0,1]
    for i in range(0,n-1):
        for j in range(i+1,n):
            if cmin > c_dist[i][j]:
                pos = [i,j]
                cmin = c_dist[i][j]
    return pos


def compute_cluster_pair_dist(list1, list2, p_dist):
    list1_num = len(list1)
    list2_num = len(list2)
    
    min_dist = p_dist[list1[0]][list2[0]]
    for i in range(0,list1_num):
        for j in range(0,list2_num):
            tem = p_dist[list1[i]][list2[j]]
            if tem < min_dist:
                min_dist = tem
                        
    return min_dist
        

def compute_cdist(board_index,cluster_list,p_dist):
    n = len(board_index)
    board = numpy.zeros((n,n))
    
    for i in range(0,n-1):
        for j in range(i+1,n):
            temp = compute_cluster_pair_dist(cluster_list[board_index[i]], cluster_list[board_index[j]], p_dist)
            board[i][j] = temp
            board[j][i] = temp
    return board
            
                        
def hierarchical_clustering(p_dist):
    global cluster_list
    cluster_list = []

    for i in range(0,gene_num):
        cluster_list.append([i])

    board_index = list(range(0,gene_num))

    res = []
    res.append(range(0,gene_num))
    c_dist = p_dist
    new_index = -1
    new_cluster = []
    pos = []
    
    for round in range (1,gene_num):    
        pos = get_min_pos(c_dist,len(c_dist))
        new_cluster = []
        new_cluster =cluster_list[board_index[pos[0]]] + cluster_list[board_index[pos[1]]]
               
        new_index = len(cluster_list)
        cluster_list.append(new_cluster)
               
        del board_index[pos[0]]
        del board_index[pos[1]-1]
        board_index.append(new_index)
        res.append(board_index[:])

        c_dist = compute_cdist(board_index,cluster_list,p_dist)
   
    return res


def cluster_labels(cluster_num,res):
    global cluster_list
    dims = numpy.matrix(res).shape
    gene_num = dims[1]
    res_n = res[gene_num-cluster_num]
    
    labels = [0]*gene_num
    for i in range(1,cluster_num+1):
        print ("\nCluster %d has genes: " % i)
        for x in cluster_list[res_n[i-1]]:
            print (x)
            labels[x] = i

    return labels


def validation(ground_truth, clustering):
    ground_truth = numpy.array(ground_truth)
    clustering = numpy.array(clustering)
    choose_index = numpy.where(ground_truth != -1)[0]
    ground_truth = ground_truth[choose_index]
    clustering = clustering[choose_index]
    
    n = len(ground_truth)
    M_gt = numpy.ones((n,n))
    M_clt = numpy.ones((n,n))
    res = 0

    for i in range(0,n-1):
        for j in range(i+1,n):
            if ground_truth[i] !=  ground_truth[j]:
                M_gt[i][j] = 0
                M_gt[j][i] = 0

            if clustering[i] != clustering[j]:
                M_clt[i][j] = 0
                M_clt[j][i] = 0
                
    M_add = M_gt + M_clt
    M_sub = M_gt - M_clt

    M11 = (M_add == 2).sum()
    M00 = (M_add == 0).sum()

    M10 = (M_sub == -1).sum()
    M01 = (M_sub == 1).sum()

    res = 1.0 * M11 / (M11+M10+M01 ) # jaccard coefficient
    #res = 1.0 * (M11+M00 ) / (M11+M00+M10+M01 )  # rand index

    return res

def pca_plot(data,label):
    pca = mypca(n_components=2)
    data = numpy.matrix(data).T
    pca.fit(data)
    data_pca = pca.components_
    fig = plt.figure()
    title = "cho: "+' Hierarchical Clustering. '+" no. of clusters- "+ str(cluster_num)
    ax = fig.gca()
    ax.scatter(data_pca[0,],data_pca[1,], c=label, marker='o',s=20)
    ax.set_title(title)
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    
    plt.grid()
    plt.show()
    return pca


gts= []
gas = []
cluster_list = []

filename = "/Users/gill/Documents/project2/code/mapreduce/new_dataset_2.txt"  # input filename

cluster_num = 3   # input number of clusters

mydata = read(filename)
gts = mydata.ground_truth
gas = mydata.gene_att
p_dist = pdist(gas, 'euclidean')
p_dist = squareform(p_dist)

all_h = hierarchical_clustering(p_dist)
labels = cluster_labels(cluster_num,all_h)

ext_res = validation(gts,labels)
print ('\nJaccard Coefficent is %f\n' % ext_res)
#print ('\nRAND index is %f\n' % ext_res) #Change the formula for calculating in validation function accordingly
print("gas",labels)
pca = pca_plot(gas,labels)