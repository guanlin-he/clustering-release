import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import argparse


# Cmd line parsing -----------------------------------------------------------
def cmdLineParsing():
  # Parser configuration
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataname", help="data name")
  parser.add_argument("--centfile", help="centroids filename")
  parser.add_argument("--algo", help="algorithm for generating the centroids (random sampling, d² sampling, k-means, k-means++)")
  parser.add_argument("--markersize", help="markersize (default: 1)", default=1.0, type=float)

  # Parsing
  args = parser.parse_args()

  # Error checking
  if args.dataname == None:
    sys.exit("Error: data name must be defined by adding --dataname")
  if args.centfile == None:
    sys.exit("Error: centroids filename must be defined by adding --centfile")
  if args.algo == None:
    sys.exit("Error: algo for generating the centroids must be defined by adding --algo")
  if args.algo not in ("rs", "d2s", "km", "kmpp"):
    sys.exit("Error: invalid algo (must be 'rs' or 'd2s' or 'km' or 'kmpp'!")

  # Argument translation
  # if args.algo == "rs":
    # args.algo = "randomsampling"
  # if args.algo == "d2s":
    # args.algo = "d2sampling"
  # if args.algo == "km":
    # args.algo = "kmeans"
  # if args.algo == "kmpp":
    # args.algo = "kmeanspp"
    
  # Returns the arg list
  return(args.dataname, args.centfile, args.algo, args.markersize)


# Parse the command line
data_name, centroids_file, algorithm, ms = cmdLineParsing()

# Read the file of centroids
X = np.loadtxt("../output/%s" % centroids_file)  

# Get the number of centroids
kc = len(X)

# Plot centroids and save the figure
plt.plot(X[:,0], X[:,1], 'o', color = 'k', markersize = ms)
plt.savefig('%s_%dcentroids_%s.png' % (data_name, kc, algorithm), dpi = 400)
# plt.show()