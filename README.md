# quixotic-autodiff

Autodidactic repository for me learning
1. c++ (or C, mostly..)
2. cuda
3. deep learning from scratch.

Currently consists of 2 reverse accumulation auto differentiation libraries using cuda and written in c++.

The first, silly-autodiff, technically works, but its a very silly implementation and is quite slow.

The second, daft-autodiff, has a lot of daft implementation choices (keeps my silly implementation of a convolution but adds to it some messiness with pointers) but is much faster.  Both are very, very slow.  Both are complete enough so that one can write a CNN or forward fee neural network model with it.

## daft-autodiff

This autodiff works by letting you describe some operations (namely, matrix products, scalar multiplications, leaky ReLUs, max pooling, and single-channel convolution) into a function, then "compile" (really just allocate memory on the gpy to compute that function) it so that one can specify values, compute the function, and compute some of its gradients.

It also supports a "batchCompute", you might think this is batch processing, id est, computing the value of the function on several input points at once, but beware! It's actually just a function to pre-load a bunch of input values all in at once, and to get the results back all in a vector of vectors.  There is the ability to pass in a helper function to run after every sample for, exempli gratia, stochastic gradient descent.  This minimizes how many times you have to copy between the device and the host.

The implementation of the convolution operation is a bit, well, daft.  It allows you to specify the kernel as a matrix, but it "unrolls" it into a larger matrix so that we can treat convolution as just a regular matrix operation.  The rolling and unrolling for the computation and gradients is probably unnecessary.

you can dive right into the code at daft_autodiff/daft_autodiff.cu .

An example of using it to implement a forward feed neural network is at examples/ffnn/daft-ffnn.cu

And an example of using it to implement LeNet-5 by using the batchCompute is at examples/lenet/daft-lenet.cu.

The test suite, tests/daft_autodiff.test.cu , is also instructive. While I avoided LLMs for most of this project, I did use them on bits I considered boring or where I felt I wasn't learning anything.  In particular, I used them on the test suite after I had written it for silly-autodiff.  Other places include the cmake configuration.

## building

### requirements

- cuda / cuda compatible gpu.  I used nvcc 12.3.r_12/3
- g++ 12.3.0 (I used spack to get mine)
- cmake
- ??? I dunno I haven't ran it on another system yet.

### building

The project supports different build configurations:

```bash
# Build in Release mode (default - maximum optimization):
cmake -DCMAKE_BUILD_TYPE=Release -B build
cmake --build ./build

# Build in Debug mode (no optimization, full debug symbols):
cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build ./build

# Build with debug info + moderate optimization (good for profiling):
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B ./build
cmake --build ./build
```

**Release mode** includes aggressive optimizations (`-O3`, `--use_fast_math`, `-march=native`) for both CPU and GPU code, which can significantly improve training performance.

**Debug mode** disables optimizations and includes full debug symbols for easier debugging with tools like `gdb` or `compute-sanitizer`.

Tests are unlikely to pass if you build in "Release mode".

I noticed on 20250627 after upgraded to glibc 2.40-26.fc41 that I was no longer able to compile the test suite.
I was getting errors like this
concurrence.h:252:32: error: cannot convert '<brace-enclosed initializer list>' to 'unsigned int' in initialization
  252 |     __gthread_cond_t _M_cond = __GTHREAD_COND_INIT;

using `dnf downgrade glibc` to downgrade to 2.40-3.fc41 seemed to have fixed the problem.

there are similar bug reports on glibc 2.41.  e.g. https://sourceware.org/bugzilla/show_bug.cgi?id=32621

It would probably be good to build it inside of a docker container for consistency on these system libraries.

## running

after building you should be able to `cd build/` and do things like `./lenet` or `./daft-lenet`.  The test suite is available by running `ctest`.

This is a very slow autodiff.  On my machine (NVIDIA GeForce GTX 1060 6gb, and an old intel cpu) daft-lenet with batch compute takes ~1400 seconds for one epoch.  Without batch compute (more memory copying) it takes ~2700 seconds.  The first "silly" implementation in lenet takes ~3400 seconds.  By comparison, a slightly simply pytorch implementation trains one epoch in about ~20s.

## What's next

- [ ] supporting batch processing - I want to figure out how to get it within 10x of a pytorch implementation!
- [ ] saving and loading model weights, maybe supporting ONNX?
- [ ] implementing support for backprop through time
- [ ] maybe some form of a repl? 
- [ ] maybe something GPUy and autodiffy that's not deep learning?

# older notes...

## silly-autodiff

A small, ad hoc, informally-specified, bug-ridden, slow implementation of half of a tensorial (is that even a word?) automatic differentiation engine written in c++/cuda.  It is complete enough so that one can write CNN or forward feed neural network models with it.

Example:
```c++
cublasHandle_t cublasH;
cublasCreate(&cublasH);

Col *x = new Col("x", 2);
x-.loadValues({1,2});

Col *y = new Col("y", 2);
y->loadValues({3, 4});

InnerProduct *f = InnerProduct(x, y);
f->computeGrad(&cublasH);

float *f_by_x = new float[2];
x->getPartials(f_by_x);

cout << "df/dx_0 = " << f_by_x[0] << ", df/dx_1 = " << f_by_x[1] << "\n";
// outputs df/dx_0 = 3, df/dx_1 = 4 
```

### features / design

There are a few basic classes designed to input data/weights, e.g. Matrix and Col. Otherwise, descendants of the main AD class are meant to be operations on matrices or column vectors. Matrices are always entered in in row-first format. E.g., if you load the values {0,1,2,3} into a matrix then you will have a 2x2 matrix whose first row is {0,1}. If you loaded that into a `Col` then you'd have a 4d column vector.

Some of those operations include:
- 1d and 2d tensors, no more.
- uses CUDA or CUBLAS wherever possible, regardless of whether or not it makes sense.
- Scalar Multiplication, requires a tensor and a float (the scalar).
- Inner Products of two column vectors.
- Matrix and Column Vector multiplication.
- Applying a Leaky ReLU to the whole tensor.
- Flatten-ing a matrix into a column vector. E.g. {0,1,2,3} as 2x2 matrix becomes a 4d column vector.
- Concatenating multiple column vectors into a larger, longer one.
- Convolution of a matrix input with a matrix kernel. 

Each operation results in a new tensor, which can be the input for the next operation, e.g.
```c++
AD* f = InnerProduct(new Col("x", 2), MatrixColProduct(new Matrix("M",2,2), new Col("y", 2)));
```
Would give you a function composed of the inner product of x with the matrix product My.
Calling `f->computeGrad(&cublasH)` would push the partially computed gradient down to the inputs "x", "y", and "M", so if you were to `->getPartials` on any of those you would have `df/d{x,y,M}`.

You'll notice that a cublas handle is always required.  When possible, we try to use cublas to perform the computations, as it's faster than I can write by hand (despite my best efforts to catch up!).
Cublas does things with columns-first, so you'll see a lot of `CUBLAS_OP_T` in the code.

The Convolution operation is done by noticing that each element of the product is actually just a dot product with the kernel. With this observation, one can write down a matrix for the kernel that, when multiplied by a column vector for the input matrix, will give you the convolution product. Written this way, one can use cublas to perform the matrix multiplication. That does mean that we've written some funky cuda kernels to convert the kernel into such a matrix, et cetera.

Because we are doing reverse accumulation, f will "push down" a seed gradient to "x" and "My", and "My" will push it down to "M" and "y". Each AD class is responsible for cleaning up the memory of whatever float* of seeds it gets from things further up the computational DAG. When we delete f, the reverse happens, and f cleans up each of its descendants, so delete f will clean up the Col and MatrixProduct.
This does mean you can destroy the universe by doing something like
```c++
Col *x = new Col("x",1);
AD *y = new MatrixColProduct(m, x);
InnerProduct *f = new InnerProduct(x, y); // note that x appears twice in the DAG
delete f; // this will try to delete x twice, causes existential problems.
```

## future work

- [ ] make it less slow
- [ ] implement AlexNet and hope it is not too slow
- [ ] implement a LTSM
- [ ] write a lisp repl for interacting with all these AD objects because compiling and stuff is annoying
- [ ] dagnabit I'll just use pytorch
- [ ] transform me maybe?


