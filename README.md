# Clustering-Algorithms
Implementation of various clustering algorithms, namely K-means, Hierarchical Agglomerative Clustering, Density Based Scanning and MapReduce K-means on Hadoop.

Readme for running the various codes:
1. For K-means Algorithm: Run the kmeans.py file from the command prompt with the python command, which will look as python means.py. The location of the input file (dataset), the initial centroids, the maximum number of iterations and the number of k clusters can be modified from within the python code.

2. For Hierarchical Agglomerative Clustering: Run the hac.py file from the command prompt with the python command, which will look as python has.py. The location of the input file and the number of k clusters can be modified from within the python code.

3. For DBScan Algorithm: Run the dbscan.py file from the command prompt with the python command, which will look as python dbscan.py. The location of the input file (dataset), the initial parameters, that is the epsilon and minimum points, and the number of k clusters can be modified from within the python code.

4. For MapReduce K-means: Directly run the hadoop.py file from within the mapreduce folder  from the command prompt, which will look as python hadoop.py. The location of the input dataset, the number k clusters, the number of mappers and reducers and the output directory of the hadoop reducer can all be configured from within the python code. The mapper.py includes all the mapper’s functions and the reducer.py includes the reducer’s functions.

All the dataset input files need to be in the same directory as the code files particularly the data files for the MapReduce K-means.
