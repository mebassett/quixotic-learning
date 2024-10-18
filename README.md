# silly-autodiff

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

## purpose

Please don't use this for anything, it's mostly for autodidactic reasons. In particular, I want to
- learn c++
- learn cuda
- learn what it takes to write the really fast/memory efficient code needed to train large deep learning models.

So I started from the ground up with a reverse accumulation autodiff.  It is very slow.  The LeNet example takes about 19 hours to train MNIST data on my machine.

## requirements

- cuda / cuda compatible gpu.  I used nvcc 12.3.r_12/3
- g++ 12.3.0 (I used spack to get mine)
- cmake
- ??? I dunno I haven't ran it on another system yet.

## building

```bash
$ cmake . -B build/
$ cmake --build build/
```

This should build both examples and the tests.  You can run the tests by
```bash
$ cd build && ctest
```
Note that the tests run twice - the second time the whole test suite is run under `compute-sanitizer --tool memcheck`, which checks for memory leaks on the GPU.

## examples

1. The tests aren't a bad place to look, see [tests/silly_autodiff.test.cu](https://gitlab.com/mebassett/quixotic-learning/-/blob/master/tests/silly_autodiff.test.cu?ref_type=heads).

2. Forward feed neural network to classify MNIST data, see [examples/ffnn/ffnn.cu](https://gitlab.com/mebassett/quixotic-learning/-/blob/master/examples/ffnn/ffnn.cu?ref_type=heads).
   After building, you should be able to run it with `build/ffnn`.

3. LeNet, a CNN to classify the same, see [examples/lenet/lenet.cu](https://gitlab.com/mebassett/quixotic-learning/-/blob/master/examples/lenet/lenet.cu?ref_type=heads).
   After building, you should be able to run it with `build/lenet`.

## features / design

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







