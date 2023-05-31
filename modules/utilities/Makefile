include ../../Makefile_definitions

CC_SOURCES = arguments.cc attachment.cc dataIO.cc feature_scaling_cpu.cc print.cc
CUDA_SOURCES = feature_scaling_gpu.cu init_gpu.cu transfers.cu transpose.cu
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

all:
	$(CPUCC) -c $(CXXFLAGS) $(CC_CXXFLAGS) $(CC_SOURCES)
	$(GPUCC) -dc $(CXXFLAGS) $(CUDA_CXXFLAGS) $(CUDA_SOURCES)
	rm -f -r cc_objects_dir
	rm -f -r cuda_objects_dir
	mkdir cc_objects_dir
	mkdir cuda_objects_dir
	mv $(CC_OBJECTS) cc_objects_dir/
	mv $(CUDA_OBJECTS) cuda_objects_dir/

clean:
	rm -f *.o *.linkinfo *~ *.bak -r cc_objects_dir -r cuda_objects_dir .depend

clear: clean


#Regles automatiques pour les objets
#%.o:  %.cc
#   $(CPUCC)  -c  $(CXXFLAGS) $(CC_CXXFLAGS) $<
#
#%.o:  %.cu
#   $(GPUCC)  -c  $(CXXFLAGS) $(CUDA_CXXFLAGS) $<

