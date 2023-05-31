import numpy as np
import matplotlib        # uncomment when running on Linux instead of Windows
matplotlib.use('Agg')    # uncomment when running on Linux instead of Windows
import matplotlib.pyplot as plt
import sys
import argparse


# Cmd line parsing -----------------------------------------------------------
def cmdLineParsing():
  # Parser configuration
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataname", help="data name")
  parser.add_argument("--labelofs", help="label offset (default: 0)", default=0, type=int)
  parser.add_argument("--markersize", help="markersize (default: 1)", default=1.0, type=float)

  # Parsing
  args = parser.parse_args()

  # Error checking
  if args.dataname == None:
    sys.exit("Error: data name must be defined by adding --dataname")
    
  # Returns the arg list
  return(args.dataname, args.labelofs, args.markersize)


# Parse the command line
data_name, labeloffset, ms = cmdLineParsing()

# Read the data file, the file of reference cluster labels (and possibly the file of centroids)
X = np.loadtxt("path-to-the-file/DATA_%s.txt" % data_name)
labels = np.loadtxt("path-to-the-file/REF_%s.txt" % data_name)
# centers = np.loadtxt("FinalCentroids.txt")

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

# Plot reference clustering
for k in range(-1, len(unique_labels)):
    class_member_mask = (labels == k + labeloffset)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', color = colors[k], markersize = ms)

# Plot centroids
# plt.plot(centers[:,0], centers[:,1], 'X', color = 'k', markersize=8)
    
# Save the figure
# plt.title('%s, %d clusters' % (data_name, len(unique_labels)))
plt.savefig('%s_clustering_ref.png' % (data_name), dpi = 400)
# plt.show()
