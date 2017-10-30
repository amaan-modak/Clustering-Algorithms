import random as rd
import math
import sys
import numpy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA as mypca
import matplotlib.pyplot as plt

intput_file = "/Users/gill/Documents/project2/code/mapreduce/new_dataset_1.txt"

numberOfCentorids= 3
max_iterations = 100
initial_centroid_ids = [2,8,19]#right now if its empty  so use line

#pairwise used for how good clusters are and how good algo is
#
def pca_plot(data,label):
    pca = mypca(n_components=2)
    data = numpy.matrix(data).T
    pca.fit(data)
    data_pca = pca.components_
    fig = plt.figure()
    title = ": "+'Kmeans: '+"number of clusters= "+str(numberOfCentorids)
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

def is_centroidSame(a, b):
	if(len(a) != len(b)):
		return 0
	for i in range(0, len(a)):#length here is 0 to number of centorid clusters
		for j in range(0, len(a[0])):# length here is number of element in each array
			if(a[i][j] != b[i][j]):
				return 0 # this will just exit the loop if executed
		return 1 #this will just exit the loop if executed

#is_centroidSame([[0.0, 1.0, 1.0],[0.0, 1.0, 2.0]], [[0.0, 1.0, 1.0],[0.0, 1.0, 2.0]])
def euclidean(a, b):
	distance = 0
	for i in range(0, len(a)):
		distance += ((b[i] - a[i])) ** 2
	final_euc= math.sqrt(distance)
	return final_euc


initial_centroids = []
data = []
real_labels = []

with open(intput_file) as inf:
	for line in inf:
		parts = line.split('\t')
		for i in range(0, len(parts)):
			parts[i] = float(parts[i])
		data.append(parts[2:len(parts)])
		real_labels.append(parts[1])
		if parts[0] in initial_centroid_ids:
			initial_centroids.append(parts[2:len(parts)])

def kmeans():
	rands = []
	new_rand = []
	centroid = []
	if len(initial_centroids) == 0:
		centroid = rd.sample(data, numberOfCentorids)
	else:
		centroid = initial_centroids
	centroid_new = []
	centroid_index = []
	num_iter = 0
	############have to make appending only unique arrays

	while(is_centroidSame(centroid, centroid_new) == 0 and num_iter < max_iterations):
		num_iter += 1
		if len(centroid_new) > 0:
			centroid = centroid_new
		centroid_new = []
		centroid_index = []
		min_indexArray=sys.maxsize
		for i in range(0, len(data)):
			min_value= sys.maxsize
			for j in range(0, len(centroid)):
				euc= euclidean(data[i], centroid[j])
				if(euc < min_value):
					min_value= euc
					min_indexArray= j  #j is the number of centroid
			centroid_index.append(min_indexArray)
			#till here we are shuffling so that we get new Cluster that is number in particulr cluster will match the previous centorid number

		#calculate new Centorids ########
		for i in range(0, len(centroid)): 
			zeros = [0] * len(centroid[0])
			x = 0
			for j in range(0, len(centroid_index)): #len becoz
				if (i== centroid_index[j]): # this will add only those that have same index number as the index number of centroid
					x += 1
					current_point = data[j]
					for k in range(0, len(zeros)): #length is that of centroid which is 2 rn
						zeros[k] = zeros[k] + current_point[k] # basically adding clumn to get new centroids

			for l in range(0, len(zeros)):
				zeros[l] = zeros[l] / x
			
			centroid_new.append(zeros)
	print(num_iter)
	print(centroid_new)
	return centroid_index

predicted_labels = kmeans()
print(predicted_labels)



pca_plot(data, predicted_labels)
jaccard_coefficient(predicted_labels, real_labels)


