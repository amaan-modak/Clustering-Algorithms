import random as rd
import math
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA as mypca
import matplotlib.pyplot as plt
import subprocess
from subprocess import Popen, PIPE
import sys
import os
import tempfile

input_file = "iyer.txt"
numberOfCentorids= 9
centroid=[]
initial_centroid_ids = set()
real_labels = []
initial_centroids = []
data = []
decoded_lines_array=[]
max_iterations = 10
num_iter = 0
#real_labels = []

with open(input_file) as inf:
	for line in inf:
		parts = line.split('\t')
		for i in range(0, len(parts)):
			parts[i] = float(parts[i])
		real_labels.append(parts[1])
		data.append(parts[2:len(parts)])
		#real_labels.append(parts[1])
		if parts[0] in initial_centroid_ids:
			initial_centroids.append(parts[2:len(parts)])
#print("cent",(centroid.size), "#######", (data.size))
centroid = rd.sample(data, numberOfCentorids)
x = np.array(centroid)
size = x.shape

oldCentroid = np.zeros(size)

matrx= np.vstack((centroid, data))

f=open('input_matrix_hadoop.txt','w+')
np.savetxt('input_matrix_hadoop.txt', matrx, fmt='%.5f')


while(not np.array_equal(oldCentroid,centroid) and num_iter < max_iterations):
	num_iter += 1

	put = os.system("hadoop fs -put input_matrix_hadoop.txt /")

	remove = os.system("hadoop fs -rm -r /output")

	execute = os.system("hadoop jar /usr/local/Cellar/hadoop/2.8.1/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar -Dmapreduce.job.maps=9 -Dmapreduce.job.reduces=9 -files ./mapper.py,./reducer.py -mapper ./mapper.py -reducer ./reducer.py -cmdenv K=9 -input /input_matrix_hadoop.txt -output /output")

	with tempfile.TemporaryFile() as tmp:
		cat = Popen(['hadoop', 'fs', '-cat', '/output/part-*'], stdout=tmp)
		cat.wait()
		tmp.seek(0)
		lines = tmp.readlines()

	cid = []
	p_list = []
	clus_list = []
	p_list_final = []
	label = []

	for line in lines:
		decoded_lines = line.decode("ascii",errors="ignore").strip()
		#print(decoded_lines)
		part = decoded_lines.split('\t')
		cid.append(part[0])
		p_list = part[1].split(',')
		p_listi = [int(feature) for feature in p_list]
		p_list_final.append(p_listi)
		centroid_list = part[2].split(',')
		cent_listf = [float(feature) for feature in centroid_list]
		clus_list.append(cent_listf)

	oldCentroid = np.copy(centroid)

	centroid = np.array(clus_list)

	centroid_data = np.vstack((centroid, data))

	f=open('input_matrix_hadoop.txt','w+')
	np.savetxt('input_matrix_hadoop.txt', centroid_data, fmt='%.5f')

	removeFile = os.system("hdfs dfs -rm -r /input_matrix_hadoop.txt")

print (cid)
print (p_list_final)

def assigneLabels(cid,p_list_final):
	label=[]
	for i in range(0,len(p_list_final)):
		x= p_list_final[i]
		for y in x:
			y= cid[i]
			#print(y)
			label.append(y)

		#print(p_list_final[i][j])
	return label

def pca_plot(data,label):
    pca = mypca(n_components=2)
    data = np.matrix(data).T
    pca.fit(data)
    data_pca = pca.components_
    fig = plt.figure()
    title = "iyer: "+'MapReduce Kmeans: '+"number of clusters= "+str(numberOfCentorids)
    ax = fig.gca()
    ax.scatter(data_pca[0,],data_pca[1,], c=label, marker='o',s=20)
    ax.set_title(title)
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    pca.fit(data)

    plt.grid()
    plt.show()

    #plt.show()
    return pca

def calculateTruths(predicted_labels, real_labels):
	tp = 0.0
	fn = 0.0
	fp = 0.0
	tn = 0.0
	for i in range(0, len(predicted_labels)):
		for j in range(i + 1, len(predicted_labels)):
			partitionLabel = (real_labels[i] == real_labels[j])
			clusterLabel = (predicted_labels[i] == predicted_labels[j])
			if partitionLabel and clusterLabel:
				tp += 1
			elif partitionLabel and not clusterLabel:
				fn += 1
			elif not partitionLabel and clusterLabel:
				fp += 1
			else:
				tn += 1
	return tp, fn, fp, tn

def jaccard_coefficient(predicted_labels, real_labels):
	tp, fn, fp, tn = calculateTruths(predicted_labels, real_labels)
	print("jacc",tp / (tp + fn + fp))
	return (tp / (tp + fn + fp))


label = assigneLabels(cid,p_list_final)

pca_plot(data,label)
jaccard_coefficient(label, real_labels)