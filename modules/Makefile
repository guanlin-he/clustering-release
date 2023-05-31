include ../Makefile_definitions

CC_SOURCES = vars.cc
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)

all:
	$(CPUCC) -c $(CXXFLAGS) $(CC_CXXFLAGS) $(CC_SOURCES)

clean:
	rm -f *.o *.linkinfo *~ *.bak *.o .depend

clear: clean


#Regles automatiques pour les objets
#%.o:  %.cc
#   $(CPUCC)  -c  $(CXXFLAGS) $(CC_CXXFLAGS) $<
#
#%.o:  %.cu
#   $(GPUCC)  -c  $(CXXFLAGS) $(CUDA_CXXFLAGS) $<

