#ifndef _DATASETS_H
#define _DATASETS_H

/* 2D shape datasets */

#ifdef Jain
    // Source: http://cs.joensuu.fi/sipu/datasets/
    #define DATASET_NAME             "Jain"
    #define NB_POINTS                373
    #define NB_REPS                  50
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          2
    #define NB_CLUSTERS              2
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Jain.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Jain.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Compound
    // Source: http://cs.joensuu.fi/sipu/datasets/
    #define DATASET_NAME             "Compound"
    #define NB_POINTS                399
    #define NB_REPS                  50
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          6
    #define NB_CLUSTERS              6
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Compound.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Compound.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Aggregation
    // Source: http://cs.joensuu.fi/sipu/datasets/
    #define DATASET_NAME             "Aggregation"
    #define NB_POINTS                788
    #define NB_REPS                  50
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          15
    #define NB_CLUSTERS              7
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Aggregation.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Aggregation.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Aggregation_A1E5
    // Source: http://cs.joensuu.fi/sipu/datasets/
    #define DATASET_NAME             "Aggregation_A1E5"
    #define NB_POINTS                78800000
    #define NB_REPS                  100000
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          15
    #define NB_CLUSTERS              7
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Aggregation_A1E5_F2E-2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Aggregation_A1E5_F2E-2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Spiral_Clean
    // Source: https://www.lancaster.ac.uk/pg/hyder/Downloads/downloads.html
    #define DATASET_NAME             "Spiral_Clean"
    #define NB_POINTS                7500
    #define NB_REPS                  500
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          6
    #define NB_CLUSTERS              3
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Spiral_Clean.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Spiral_Clean.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Spirals_A1E2
    // Source: amplification on https://www.lancaster.ac.uk/pg/hyder/Downloads/downloads.html
    #define DATASET_NAME             "Spirals_A1E2"
    #define NB_POINTS                750000
    #define NB_REPS                  500
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          6
    #define NB_CLUSTERS              3
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Spiral_Clean_A1E2_F1E-2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Spiral_Clean_A1E2_F1E-2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif


#ifdef Spirals_A1E3
    // Source: amplification on https://www.lancaster.ac.uk/pg/hyder/Downloads/downloads.html
    #define DATASET_NAME             "Spirals_A1E3"
    #define NB_POINTS                7500000
    #define NB_REPS                  500
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          6
    #define NB_CLUSTERS              3
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Spiral_Clean_A1E3_F1E-2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Spiral_Clean_A1E3_F1E-2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Spirals_A1E4
    // Source: amplification on https://www.lancaster.ac.uk/pg/hyder/Downloads/downloads.html
    #define DATASET_NAME             "Spirals_A1E4"
    #define NB_POINTS                75000000
    #define NB_REPS                  1000
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          6
    #define NB_CLUSTERS              3
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Spiral_Clean_A1E4_F1E-2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Spiral_Clean_A1E4_F1E-2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Smile2
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/smile2.arff
    #define DATASET_NAME             "Smile2"
    #define NB_POINTS                1000
    #define NB_REPS                  50
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          4
    #define NB_CLUSTERS              4
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Smile2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Smile2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Smile2_A1E5
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/smile2.arff
    #define DATASET_NAME             "Smile2_A1E5"
    #define NB_POINTS                100000000
    #define NB_REPS                  1000000
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          4
    #define NB_CLUSTERS              4
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Smile2_A1E5_F5E-2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Smile2_A1E5_F5E-2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef S1
    // Source: http://cs.joensuu.fi/sipu/datasets/
    #define DATASET_NAME             "S1"
    #define NB_POINTS                5000
    #define NB_REPS                  500
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          20
    #define NB_CLUSTERS              15
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_S1.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_S1.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef S4
    // Source: http://cs.joensuu.fi/sipu/datasets/
    #define DATASET_NAME             "S4"
    #define NB_POINTS                5000
    #define NB_REPS                  500
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          20
    #define NB_CLUSTERS              15
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_S4.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_S4.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Cure_t1
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/cure-t1-2000n-2D.arff
    #define DATASET_NAME             "Cure_t1"
    #define NB_POINTS                2000
    #define NB_REPS                  50
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          6
    #define NB_CLUSTERS              6
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Cure_t1.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Cure_t1.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Cure_t2  // Similar to Cure_t1 but with some noise
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/cure-t2-4k.arff
    #define DATASET_NAME             "Cure_t2"
    #define NB_POINTS                4200   // NbNoise = 200
    #define NB_REPS                  50
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          7
    #define NB_CLUSTERS              5  // 7
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Cure_t2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Cure_t2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Complex8
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/complex9.arff
    #define DATASET_NAME             "Complex8"
    #define NB_POINTS                2551
    #define NB_REPS                  300
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          8
    #define NB_CLUSTERS              8
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Complex8.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Complex8.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Cluto_t8  // Similar to Complex8 but with some noise
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/cluto-t8-8k.arff
    #define DATASET_NAME             "Cluto_t8"
    #define NB_POINTS                8000   // NbNoise = 323
    #define NB_REPS                  500
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          9
    #define NB_CLUSTERS              9
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Cluto_t8.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Cluto_t8.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Complex9
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/complex9.arff
    #define DATASET_NAME             "Complex9"
    #define NB_POINTS                3031
    #define NB_REPS                  3
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          9
    #define NB_CLUSTERS              9
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Complex9.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Complex9.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Complex9_A1E5
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/complex9.arff
    #define DATASET_NAME             "Complex9_A1E5"
    #define NB_POINTS                303100000
    #define NB_REPS                  1000000
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          9
    #define NB_CLUSTERS              9
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Complex9_A1E5_F1E-2.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Complex9_A1E5_F1E-2.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Cluto_t7  // Similar to Complex9 but with some noise
    // Source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/cluto-t7-10k.arff
    #define DATASET_NAME             "Cluto_t7"
    #define NB_POINTS                10000  // NbNoise = 792
    #define NB_REPS                  500
    #define NB_DIMS                  2
    #define MAX_NB_CLUSTERS          15
    #define NB_CLUSTERS              10
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Cluto_t7.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Cluto_t7.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
    // NVGRAPH_STATUS_INTERNAL_ERROR ./Clustering -gpu-s 3 -sim-metric 2 -sigma 0.015 -thold-sim 0.15 -filter-noise-sol 1 -thold-noise 0.2
#endif


/* MNIST-based datasets */

#ifdef MNIST
    // Source: http://yann.lecun.com/exdb/mnist/
    #define DATASET_NAME             "MNIST"
    #define NB_POINTS                60000
    #define NB_REPS                  600
    #define NB_DIMS                  784
    #define MAX_NB_CLUSTERS          10
    #define NB_CLUSTERS              10
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_MNIST_train.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_MNIST_train.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef MNIST120K
    // Source: https://leon.bottou.org/projects/infimnist
    #define DATASET_NAME             "MNIST120K"
    #define NB_POINTS                120000
    #define NB_REPS                  600
    #define NB_DIMS                  784
    #define MAX_NB_CLUSTERS          10
    #define NB_CLUSTERS              10
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_MNIST120K.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_MNIST120K.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef MNIST240K
    // Source: https://leon.bottou.org/projects/infimnist
    #define DATASET_NAME             "MNIST240K"
    #define NB_POINTS                240000
    #define NB_REPS                  600
    #define NB_DIMS                  784
    #define MAX_NB_CLUSTERS          10
    #define NB_CLUSTERS              10
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_MNIST240K.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_MNIST240K.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef US_Census_1990
    // Source: https://archive.ics.uci.edu/ml/datasets/US+Census+data+(1990)
    #define DATASET_NAME             "US_Census_1990"
    #define NB_POINTS                2458285
    #define NB_REPS                  300
    #define NB_DIMS                  68
    #define MAX_NB_CLUSTERS          16
    #define NB_CLUSTERS              16
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_USCensus1990.txt"
    #define INPUT_REF_LABELS         ""
    #define INPUT_INITIAL_CENTROIDS  "/data/data_he/Benchmark_datasets/InitialCentroids_DATA_USCensus1990_16clusters.txt"
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Clouds4D_1E6
    // Source: https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans/-/blob/master/Synthetic_Data_Generator.py
    #define DATASET_NAME             "Clouds4D_1E6"
    #define NB_POINTS                1000000
    #define NB_REPS                  300
    #define NB_DIMS                  4
    #define MAX_NB_CLUSTERS          4
    #define NB_CLUSTERS              4
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Clouds4D_1E6_shuffled.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Clouds4D_1E6_shuffled.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Clouds4D_5E6
    // Source: https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans/-/blob/master/Synthetic_Data_Generator.py
    #define DATASET_NAME             "Clouds4D_5E6"
    #define NB_POINTS                5000000
    #define NB_REPS                  300
    #define NB_DIMS                  4
    #define MAX_NB_CLUSTERS          4
    #define NB_CLUSTERS              4
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/DATA_Clouds4D_5E6_sorted.txt"
    #define INPUT_REF_LABELS         "/data/data_he/Benchmark_datasets/REF_Clouds4D_5E6_sorted.txt"
    #define INPUT_INITIAL_CENTROIDS  ""
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#ifdef Clouds4D_5E7
    // Source: https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans/-/blob/master/Synthetic_Data_Generator.py
    #define DATASET_NAME             "Clouds4D_5E7"
    #define NB_POINTS                50000000
    #define NB_REPS                  300
    #define NB_DIMS                  4
    #define MAX_NB_CLUSTERS          4
    #define NB_CLUSTERS              4
    #define INPUT_DATA               "/data/data_he/Benchmark_datasets/InputDataset-50million.txt"
    #define INPUT_REF_LABELS         ""
    #define INPUT_INITIAL_CENTROIDS  "/data/data_he/Benchmark_datasets/InitialCentroids_InputDataset-50million.txt"
    #define INPUT_REF_CENTROIDS      ""
    #define INPUT_BINARY_DATA        ""
#endif

#endif