ffnn-fast-nvcc:
	nvcc -Xptxas -O3,-v -Xcompiler -O3 --std c++20 -lcublas mnistdata.cpp fast_autodiff.cu ffnn-fast_autodiff.cu -o fast-ffnn-nvcc

test: 
	nvcc -lineinfo --std c++20 -lcublas fast_autodiff.cu test_fast_autodiff.cu -o test

lenet:
	nvcc -Xptxas -O3,-v -Xcompiler -O3 --std c++20 -lcublas mnistdata.cpp fast_autodiff.cu lenet.cu -o lenet


