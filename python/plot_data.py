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
  parser.add_argument("--markersize", help="markersize (default: 1)", default=1.0, type=float)

  # Parsing
  args = parser.parse_args()

  # Error checking
  if args.dataname == None:
    sys.exit("Error: data name must be defined by adding --dataname")

  # Returns the arg list
  return(args.dataname, args.markersize)


# Parse the command line
data_name, ms = cmdLineParsing()

# Read the data file
X = np.loadtxt("path-to-the-file/DATA_%s.txt" % data_name)

# Plot the data
plt.plot(X[:,0], X[:,1], 'o', color = 'k', markersize = ms)

# Save the figure
# plt.title('%s' % data_name, fontsize=18)
plt.savefig('%s.png' % data_name, dpi = 400)
# plt.show()
