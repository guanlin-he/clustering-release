# -*- coding: utf-8 -*-
print(__doc__)

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_kernels
import time
import argparse
import sys


# Cmd line parsing -----------------------------------------------------------
def cmdLineParsing():
  # Parser configuration
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataname", help="data name")
  parser.add_argument("--sigma", help="sigma", default=0.01, type=float)
  parser.add_argument("--thold", help="similarity threshold", default=0.01, type=float)
  parser.add_argument("--markersize", help="markersize (default: 1)", default=1.0, type=float)

  # Parsing
  args = parser.parse_args()
  
  # Error checking
  if args.dataname == None:
    sys.exit("Error: data name must be defined by adding --dataname")
    
  # Returns the arg list
  return(args.dataname, args.sigma, args.thold, args.markersize)


# Parse the command line
data_name, sigma, thold, ms = cmdLineParsing()

# Read the file of representatives
X = np.loadtxt("../output/Representatives.txt")
print("Reps loading finished.")


# #############################################################################
# Compute adjacency matrix (i.e. similarity matrix)
t_pk_0 = time.time()
adjacency_matrix = pairwise_kernels(X, metric='rbf', gamma=1/(2*sigma*sigma)) #n_jobs=1
adjacency_matrix[adjacency_matrix <= thold] = 0
np.fill_diagonal(adjacency_matrix, 0)
t_pk_1 = time.time()
print("Pairwise kernel computation finished.")
print("Elapsed time of pairwise kernel: %0.3f s" % (t_pk_1 - t_pk_0))
# np.savetxt("matrix.txt", np.array(adjacency_matrix))


# #############################################################################
# Perform Spectral Clustering based on the adjacency matrix
t_sc_0 = time.time()
sc = SpectralClustering(n_clusters=9, eigen_solver='lobpcg', eigen_tol=1E-5,
                        random_state=1, affinity='precomputed', n_init=1) #n_jobs=1
sc.fit_predict(adjacency_matrix)
t_sc_1 = time.time()
labels_reps = sc.labels_
print("Spectral clustering finished.")
print("Elapsed time of spectral clustering: %0.3f s" % (t_sc_1 - t_sc_0))
# np.savetxt("LabelsOfReps.txt", np.array(labels_reps), fmt='%i')


# #############################################################################
# Plot representatives
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = ["green", "blue", "red", "orange", "cyan", "brown", "pink", "purple", "olive", "gray",
          "tomato", "lime", "wheat", "darkblue", "violet", "darkcyan", "maroon", "deeppink", "olivedrab", "darkgray", 
          "tomato", "lime", "cornflowerblue", "bisque", "magenta", "darkturquoise", "sandybrown", "hotpink", "darkkhaki", "silver",
          "lightcoral", "lightgreen", "dodgerblue", "wheat", "orchid", "lightseagreen", "darksalmon", "palevioletred", "gold", "dimgray",
          "green", "blue", "red", "orange", "cyan", "brown", "pink", "purple", "olive", "gray",
          "darkred", "darkgreen", "darkorange", "darkblue", "violet", "darkcyan", "maroon", "deeppink", "olivedrab", "darkgray", 
          "tomato", "lime", "cornflowerblue", "bisque", "magenta", "darkturquoise", "sandybrown", "hotpink", "darkkhaki", "silver",
          "lightcoral", "lightgreen", "dodgerblue", "wheat", "orchid", "lightseagreen", "darksalmon", "palevioletred", "gold", "dimgray",
          "green", "blue", "red", "orange", "cyan", "brown", "pink", "purple", "olive", "gray",
          "darkred", "darkgreen", "darkorange", "darkblue", "violet", "darkcyan", "maroon", "deeppink", "olivedrab", "darkgray"]
markers = ["s", "v", "o", "^", "<", ">", "p", "P", "*", "X", 
           "D", "1", "d", "2", "+", "3", "|", "4", "x", "_",
           "X", "*", "P", "p", ">", "o", "s", "v", "^", "<",
           "_", "3", "|", "x", "+", "4", "D", "2",  "d", "1",
           "_", "3", "|", "x", "+", "4", "D", "2",  "d", "1",
           "X", "*", "P", "p", ">", "o", "s", "v", "^", "<",
           "D", "1", "d", "2", "+", "3", "|", "4", "x", "_",
           "s", "v", "o", "^", "<", ">", "p", "P", "*", "X", 
           "D", "1", "d", "2", "+", "3", "|", "4", "x", "_",
           "_", "3", "|", "x", "+", "4", "D", "2",  "d", "1"]
unique_labels = set(labels_reps)
i = 0
for k in unique_labels:
    class_member_mask = (labels_reps == k)

    xy = X[class_member_mask]
    if k == -1:
        plt.plot(xy[:, 0], xy[:, 1], 'o', color = colors[-1], markersize = ms)
    else:
        plt.plot(xy[:, 0], xy[:, 1], 'o', color = colors[i], markersize = ms)
    i = i + 1

plt.title('SC on %d reps (sigma=%f, simThold=%f)' % (len(X), sigma, thold))
plt.savefig('sc_on_reps.png', dpi = 400)

sys.exit()


# #############################################################################
# Input data attachment
labels_all_data = np.loadtxt("../output/LabelsToReps.txt", dtype=int)   # uncomment when running on Linux instead of Windows
t_da_0 = time.time()
labels_all_data = labels_reps[labels_all_data]
t_da_1 = time.time()
print("Input data attachment finished.")
print("Elapsed time of input data attachment: %0.3f s" % (t_da_1 - t_da_0))
# np.savetxt("Labels.txt", np.array(labels_all_data), fmt='%i')


# #############################################################################
# Evaluate clustering quality
from sklearn import metrics
labels_true = np.loadtxt("path-to-the-file/REF_%s.txt" % data_name, dtype=int)
t_ev_0 = time.time()
ARI = metrics.adjusted_rand_score(labels_true, labels_all_data)
AMI = metrics.adjusted_mutual_info_score(labels_true, labels_all_data) 
NMI = metrics.normalized_mutual_info_score(labels_true, labels_all_data)
t_ev_1 = time.time()
print("Clustering quality evaluation finished.")
print("Elapsed time of clustering quality evaluation: %0.3f s" % (t_ev_1 - t_ev_0))
print("ARI = %.3f" % ARI)
print("AMI = %.3f" % AMI)
print("NMI = %.3f" % NMI)