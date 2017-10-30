#!/usr/bin/env python3

import random as rd
import math
import sys
import numpy as np

sum = []
count = 0

point_list = []
oldCluster = None

for line in sys.stdin:
	points= [x for x in line.split()]
	cluster_index=int(points[0])

	if oldCluster == None:
		oldCluster=int(points[0])
		point_list.append(int(points[1]))
		#print (points[2:])
		scores = points[2].strip().split(',')
		sum = np.array(scores, dtype=float)
		count +=1

	elif cluster_index == oldCluster:
		point_list.append(int(points[1]))
		scores = points[2].strip().split(',')
		#print (scores)
		sum += np.array(scores, dtype=float)
		count += 1

	elif cluster_index != oldCluster:
		mean = sum/count
		print (oldCluster, "\t", point_list, "\t", ",".join([str(feature) for feature in mean]))
		count = 1		
		point_list = []
		point_list.append(int(points[1]))
		scores = points[2].strip().split(',')
		sum = np.array(scores, dtype=float)
		oldCluster = cluster_index

mean=sum/count
print ("%s\t%s\t%s" %(cluster_index, ",".join([str(feature) for feature in point_list]), ",".join([str(feature) for feature in mean])))