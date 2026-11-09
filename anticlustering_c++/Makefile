CXX ?= g++
MATLAB_DIR ?= /path/to/matlab
GUROBI_DIR ?= /path/to/gurobi
GUROBI_LIB ?= gurobi110

TARGET := cap
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)

CXXFLAGS ?= -O3 -march=native -ffast-math -DNDEBUG -DARMA_NO_DEBUG
CPPFLAGS += -I$(MATLAB_DIR)/extern/include -I$(GUROBI_DIR)/include
LDFLAGS += -L$(MATLAB_DIR)/extern/bin/glnxa64 -L$(GUROBI_DIR)/lib
LDLIBS += -larmadillo -pthread -llapack -lopenblas \
          -lMatlabDataArray -lMatlabEngine -lgurobi_c++ -l$(GUROBI_LIB)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS) \
		-Wl,-rpath,$(MATLAB_DIR)/extern/bin/glnxa64 \
		-Wl,-rpath,$(GUROBI_DIR)/lib

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) $(OBJECTS) $(TARGET)
