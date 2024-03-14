fnn:
	 g++  -std=c++20  -fmodules-ts  mnistdata.cpp autodiff.cpp   fnn-autodiff.cpp  -o fnn -x c++-system-header iostream random fstream sstream cmath valarray vector

