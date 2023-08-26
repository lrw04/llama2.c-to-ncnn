CXX = c++
CXXOPTS = -std=c++17 -O2

NCNN_PATH = ~/ncnn

all: inference convert

clean:
	rm -f inference convert

inference: inference.cpp
	$(CXX) inference.cpp $(NCNN_PATH)/build/src/libncnn.a -I $(NCNN_PATH)/src -I $(NCNN_PATH)/build/src/ -o inference $(CXXOPTS) -fopenmp

convert: convert.cpp
	$(CXX) convert.cpp -o convert $(CXXOPTS)

.PHONY: all clean
