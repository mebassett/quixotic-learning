fnn:
	 g++  -std=c++20  -fmodules-ts  mnistdata.cpp autodiff.cpp   fnn-autodiff.cpp  -o fnn -x c++-system-header iostream random fstream sstream cmath valarray vector

fast-ffnn:
	 g++  -std=c++20  -fmodules-ts  mnistdata.cpp fast_autodiff.cpp ffnn-fast_autodiff.cpp  -o fast-ffnn -x c++-system-header iostream random fstream sstream cmath valarray vector

test-ad-xor:
	 g++  -std=c++20  -fmodules-ts  mnistdata.cpp autodiff.cpp   autodiff-xor-test.cpp  -o test-ad-xor -x c++-system-header iostream random fstream sstream cmath valarray vector
