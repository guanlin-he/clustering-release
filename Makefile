include Makefile_definitions

CC_SOURCES = main.cc
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)
CC_THEME_OBJECTS = modules/*.o  modules/utilities/cc_objects_dir/*.o  modules/kmeans/cc_objects_dir/*.o  modules/spectral_clustering/cc_objects_dir/*.o
CC_ALL_OBJECTS = $(CC_OBJECTS) $(CC_THEME_OBJECTS)
CUDA_ALL_OBJECTS = modules/utilities/cuda_objects_dir/*.o  modules/kmeans/cuda_objects_dir/*.o  modules/spectral_clustering/cuda_objects_dir/*.o

EXECNAME = Clustering


all:
	cd modules; make
	cd modules/utilities; make
	cd modules/kmeans; make
	cd modules/spectral_clustering; make
	$(CPUCC) -c $(CXXFLAGS) $(CC_CXXFLAGS) $(CC_SOURCES)
	$(GPUCC) -dlink $(CXXFLAGS) $(CUDA_CXXFLAGS) $(CUDA_ALL_OBJECTS) -o __gpu_rdc.o -lcudadevrt
	$(CPUCC) -o $(EXECNAME) $(CC_LDFLAGS) $(CUGRAPH_LDFLAGS) $(CUDA_LDFLAGS) $(CUDA_ALL_OBJECTS) __gpu_rdc.o $(CC_ALL_OBJECTS) $(CUGRAPH_LIBS) $(CUDA_LIBS) $(CC_LIBS)

# depend:
	# makedepend -- $(CXXFLAGS) -- $(CC_SOURCES)

clean:
	cd modules; make clean
	cd modules/utilities; make clean
	cd modules/kmeans; make clean
	cd modules/spectral_clustering; make clean
	rm -f *.o $(EXECNAME) *.linkinfo *~ *.bak .depend
	rm -f output/*.txt
	rm -f output/*.csv
	rm -f output/*.png

clear: clean


#Regles automatiques pour les objets
#%.o:  %.cc
#   $(CPUCC)  -c  $(CXXFLAGS) $(CC_CXXFLAGS) $<
#
#%.o:  %.cu
#   $(GPUCC)  -c  $(CXXFLAGS) $(CUDA_CXXFLAGS) $<

