include ../../Makefile_definitions

CC_SOURCES = constr_sim_matrix_on_cpu.cc
CUDA_SOURCES = constr_sim_lap_matrix_in_dense.cu \
               constr_epsilon_sim_matrix_in_csr.cu  constr_epsilon_sim_matrix_in_dense_to_csr.cu \
               get_edge_list.cu  filter_noise.cu  nvgraph_api.cu  normalize_eigvect_mat.cu  auto_tuning.cu \
               sc_gpu_cusolverdn.cu  sc_gpu_nvgraph.cu  sc_gpu_nvgraph_km.cu  sc_gpu_cugraph.cu
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

