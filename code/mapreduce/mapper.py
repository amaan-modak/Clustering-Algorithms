#!/usr/bin/env python3

import random as rd
import math
import sys
import numpy as np
import os

#intput_file = "/Users/gill/Documents/data_min_Pro2_submit/code_pro2/input_matrix_hadoop.txt"

def euclidean(a, b):
	distance = 0
	for i in range(0, len(a)):
		distance += ((b[i] - a[i])) ** 2
	final_euc= math.sqrt(distance)
	return final_euc

#k=2
k=int(os.environ["K"]) #number of clusters in data file
data=[]
centroids = []
feature_matrix = []

for line in sys.stdin:
	parts = line.split(' ')
	for i in range(0, len(parts)):
		parts[i] = float(parts[i])
	data.append(parts[0:len(parts)])
#print(data)

centroids = data[0:k]
feature_matrix = data[k:]

#print(centroids)
#print(feature_matrix)
euc=0
centroid_index=[]
pointIndex=[]

for i in range(0, len(feature_matrix)):
			min_value= sys.maxsize
			p= i+1
			pointIndex.append(p)
			for j in range(0, len(centroids)):
				euc= euclidean(feature_matrix[i], centroids[j])
				if(euc < min_value):
					min_value= euc
					min_indexArray= j+1  #j is the number of centroid
			centroid_index.append(min_indexArray)
			print("%s\t%s\t%s" % (min_indexArray, p, ",".join([str(feature) for feature in feature_matrix[i]])))




