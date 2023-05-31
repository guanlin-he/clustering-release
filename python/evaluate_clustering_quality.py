# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
# Synthetic datasets                                                                                                                                                              */
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/ 
# DATASET_NAME       NB_POINTS     NB_DIMS  NB_CLUSTERS  INPUT_DATA                     INPUT_REF_LABELS
# S1                 5000          2        15           DATA_S1.txt                    REF_S1.txt
# R15                600           2        15           DATA_R15.txt                   REF_R15.txt
# Aggregation        788           2        7            DATA_Aggregation.txt           REF_Aggregation.txt
# Spirals            7500          2        3            DATA_Spiral_Clean.txt          REF_Spiral_Clean.txt
# Birch1             100000        2        100          DATA_Birch1.txt                REF_Birch1.txt
# Birch2             100000        2        100          DATA_Birch2.txt                REF_Birch2.txt
# Clouds4D-1M        1000000       4        4            DATA_Clouds4D_1E6.txt          REF_Clouds4D_1E6.txt
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
# Real-world datasets                                                                                                                                                             */
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/ 
# DATASET_NAME       NB_POINTS     NB_DIMS  NB_CLUSTERS  INPUT_DATA                     INPUT_REF_LABELS                      Source                                       URL
# LetterRecognition  20000         16       26           DATA_Letter_Recognition.txt    REF_Letter_Recognition.txt            UCI ML Repository                            https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
# MNIST120K          120000        784      10           DATA_MNIST120K.txt             REF_MNIST120K.txt                     THE MNIST DATABASE of handwritten digits     http://yann.lecun.com/exdb/mnist/
# MNIST240K          240000        784      10           DATA_MNIST240K.txt             REF_MNIST240K.txt                     THE MNIST DATABASE of handwritten digits     http://yann.lecun.com/exdb/mnist/

from sklearn import metrics
import sys
import argparse


# Cmd line parsing -----------------------------------------------------------
def cmdLineParsing():
  # Parser configuration
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataname", help="data name")

  # Parsing
  args = parser.parse_args()

  # Error checking
  if args.dataname == None:
    sys.exit("Error: data name must be defined by adding --dataname")

  # Returns the arg list
  return(args.dataname)


# Parse the command line
data_name = cmdLineParsing()

# Read the file of reference cluster labels
fo = open("path-to-the-file/REF_%s.txt" % data_name, 'r')
lines_labels_true = fo.readlines()
fo.close()

# Read the file of cluster labels to be evaluated
fo = open("../output/Labels.txt", 'r')
lines_labels_clustering = fo.readlines()
fo.close()

labels_true = []
labels_clustering = []

for line in lines_labels_true:
    p = line.split()
    labels_true.append(int(p[0]))

for line in lines_labels_clustering:
    p = line.split()
    labels_clustering.append(int(p[0]))

# Compute three metrics for evaluating clustering quality
ARI = metrics.adjusted_rand_score(labels_true, labels_clustering)
AMI = metrics.adjusted_mutual_info_score(labels_true, labels_clustering) 
NMI = metrics.normalized_mutual_info_score(labels_true, labels_clustering)

# Print the results
print("ARI = %.3f" % ARI)
print("AMI = %.3f" % AMI)
print("NMI = %.3f" % NMI)
