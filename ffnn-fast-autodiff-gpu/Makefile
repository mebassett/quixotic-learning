ffnn-fast-nvcc:
	nvcc -Xptxas -O3,-v -Xcompiler -O3 --std c++20 mnistdata.cpp fast_autodiff.cu ffnn-fast_autodiff.cpp -o fast-ffnn-nvcc

test: 
	nvcc --std c++20 fast_autodiff.cu test_fast_autodiff.cu -o test
