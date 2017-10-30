import sys
import os
import math
import random
import numpy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA as mypca
import matplotlib.pyplot as plt
intput_file = "/Users/gill/Documents/project2/code/mapreduce/new_dataset_1.txt"
epsilon = 1.0
min_points = 4

data = []
real_labels = []#real labels stores
with open(intput_file) as inf:
	for line in inf:
		parts = line.split('\t')
		for i in range(0, len(parts)):
			parts[i] = float(parts[i])
		data.append(parts[2:len(parts)])
		real_labels.append(parts[1])
def pca_plot(data,label):
    pca = mypca(n_components=2)
    data = numpy.matrix(data).T
    pca.fit(data)
    data_pca = pca.components_
    fig = plt.figure()
    title = "iyer : "+'DBscan: '+"epsilon: "+str(epsilon)+"," + "min_pts: " + str(min_points)
    ax = fig.gca()
    ax.scatter(data_pca[0,],data_pca[1,], c=label, marker='o',s=20)
    ax.set_title(title)
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    
    plt.grid()
    plt.show()
    return pca

#union can only be done if cluster numbers(ground truth and our generated) are same that is why truth table is calculatedd
def calculateTruths(predicted_labels, real_labels):# to find true positive etc
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
	print((tp / (tp + fn + fp)))
	return (tp / (tp + fn + fp))


def rand_index(predicted_labels, real_labels):
	tp, fn, fp, tn = calculateTruths(predicted_labels, real_labels)
	return ((tp + tn) / (tp + fn + fp + tn))


def get_euclidean_distance(a, b):
	distance = 0
	for i in range(0, len(a)):
		distance +=  math.pow(b[i] - a[i], 2)
	return  math.pow(distance, 0.5)

assigned_labels = [-1] * len(real_labels)
print(assigned_labels)
unassigned_points_index = set(range(0, len(assigned_labels)))
current_cluster_num = 0

def find_all_reachable_points(data, unassigned_points_index, index):
	core_point_count = 0
	points_in_cluster = set()
	points_to_explore = set([index])
	point_already_explored = set()
	while(len(points_to_explore) != 0):
		c_index = points_to_explore.pop()
		point_already_explored.add(c_index)
		for p_index in unassigned_points_index:
			dist = get_euclidean_distance(data[c_index], data[p_index])
			if dist <= epsilon and p_index != c_index:
				core_point_count += 1
				if p_index not in point_already_explored:
					points_to_explore.add(p_index)
		if core_point_count >= min_points:
			for i in points_to_explore:
				points_in_cluster.add(i)
	return points_in_cluster

while(len(unassigned_points_index) != 0):
	index = random.sample(unassigned_points_index, 1)[0]
	unassigned_points_index.remove(index)
	rechable_points = find_all_reachable_points(data, unassigned_points_index, index)
	if rechable_points:
		current_cluster_num += 1
		assigned_labels[index] = current_cluster_num
	for p_index in rechable_points:
		assigned_labels[p_index] = current_cluster_num
		unassigned_points_index.remove(p_index)

#print((assigned_labels))
jaccard_coefficient(assigned_labels, real_labels)
pca=pca_plot( data ,assigned_labels)