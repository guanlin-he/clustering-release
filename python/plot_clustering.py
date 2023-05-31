import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse


# Cmd line parsing -----------------------------------------------------------
def cmdLineParsing():
  # Parser configuration
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataname", help="data name")
  parser.add_argument("--algo", help="clustering algorithm (k-means, k-means++, spectral clustering)")
  # parser.add_argument("--labelofs", help="label offset (default: 0)", default=0, type=int)
  parser.add_argument("--markersize", help="markersize (default: 1)", default=1.0, type=float)

  # Parsing
  args = parser.parse_args()

  # Error checking
  if args.dataname == None:
    sys.exit("Error: data name must be defined by adding --dataname")
  if args.algo == None:
    sys.exit("Error: clustering algorithm must be defined by adding --algo")
  if args.algo not in ("km", "kmpp", "sc"):
    sys.exit("Error: invalid algo (must be 'km' or 'kmpp' or 'sc'!")
    
  # Returns the arg list
  # return(args.dataname, args.algo, args.labelofs, args.markersize)
  return(args.dataname, args.algo, args.markersize)


# Parse the command line
data_name, algorithm, ms = cmdLineParsing()
# data_name, algorithm, labeloffset, ms = cmdLineParsing()

# Read the data file, the file of cluster labels (and possibly the file of centroids)
X = np.loadtxt("/usr/users/intercell/ic_he/Benchmark_datasets/DATA_%s.txt" % data_name)
labels = np.loadtxt("../output/Labels.txt")
# centers = np.loadtxt("../output/FinalCentroids.txt")

# Define colors and markers
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

# Find the unique labels 
unique_labels = set(labels)

# Plot clustering
i = 0
for k in unique_labels:
# for k in range(-1, len(unique_labels)):
    class_member_mask = (labels == k)
    # class_member_mask = (labels == k + labeloffset)

    xy = X[class_member_mask]
    if k == -1:
        plt.plot(xy[:, 0], xy[:, 1], 'o', color = colors[-1], markersize = ms)
    else:
        plt.plot(xy[:, 0], xy[:, 1], 'o', color = colors[i], markersize = ms)
    i = i + 1

# Plot centroids
# plt.plot(centers[:,0], centers[:,1], 'X', color = 'k', markersize=8)

# Save the figure
# plt.title('%s, %d clusters' % (data_name, len(unique_labels)))
plt.savefig('%s_clustering_%s.png' % (data_name, algorithm), dpi = 400)
# plt.show()
