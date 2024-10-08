#include <iostream>
#include <valarray>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "fast_autodiff.h"

using namespace std;
using namespace FA;

int main() {
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    cout << "Testing Scalar \n";

    Col* xy = new Col("xy", 2);

    xy->loadValues({ 1.0, 2.0});
    
    Scalar* test_scalar = new Scalar(xy, 5);
    test_scalar->compute(&cublasH);
    test_scalar->fromDevice();

    if(  test_scalar->value[0] != 5.0
      || test_scalar->value[1] != 10.0 ) {
        cout << "Scalar failed!  should be {5, 10} but it is "
             << "{" << test_scalar->value[0] << ", " << test_scalar->value[1]
             << "}.\n";

    }

    cout << "Testing InnerProduct \n";

    Col* ab = new Col("ab", 2);
    ab->loadValues({ 3.0, 4.0 });
    InnerProduct* test_ip = new InnerProduct(xy, ab);
    test_ip->compute(&cublasH);
    test_ip->fromDevice();
    

    if( *test_ip->value != 11.0) {
        cout << "InnerProduct failed!  should be 11 but it is" << *test_ip->value << ".\n";
    }

    cout << "Testing AddCol \n";

    AddCol* test_add = new AddCol(xy, ab);
    test_add->compute(&cublasH);
    test_add->fromDevice();

    if (test_add->value[0] != 4 || test_add->value[1] != 6 ) {
        cout << "AddCol failed! Should be  {4, 6} but its" 
             << "{" << test_add->value[0] << ", " << test_add->value[1] << "}\n";
    }

    cout << "Testing MatrixColProduct\n";

    float *matrixGrad = new float[4];
    Matrix* abcd = new Matrix("abcd", 2, 2);
    abcd->loadValues({1,-1,-1, 1});
    cudaMemcpy(matrixGrad, abcd->d_grad, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Matrix grad\n"
         << matrixGrad[0] << ", " << matrixGrad[1] << "\n" 
         << matrixGrad[2] << ", " << matrixGrad[3] << "\n";

    MatrixColProduct *test_matCol = new MatrixColProduct(abcd, xy);
    test_matCol->compute(&cublasH);
    test_matCol->fromDevice();
    test_matCol->computeGrad(&cublasH);

    cudaMemcpy(matrixGrad, abcd->d_grad, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Matrix grad\n"
         << matrixGrad[0] << ", " << matrixGrad[1] << "\n" 
         << matrixGrad[2] << ", " << matrixGrad[3] << "\n";
    delete matrixGrad;

    xy->fromDevice();

    cout << "xy: {" << xy->value[0] << ", " << xy->value[1] << "}\n";


    if (test_matCol->value[0] != -1 || test_matCol->value[1] != 1 ) {
        cout << "MatrixColProduct failed! Should be  {-1, 1} but its" 
             << "{" << test_matCol->value[0] << ", " << test_matCol->value[1] << "}\n";
    }

    cout << "Testing Leaky ReLU\n";

    Col* z = new Col("z", 4);
    ColLeakyReLU* relu = new ColLeakyReLU(z);
    z->loadValues({500, -500, 0.5, -1});
    relu->compute(&cublasH);
    relu->fromDevice();

    cout << relu->value[0] << ", " << relu->value[1] << ", " << relu->value[2] << ", " << relu->value[3]
         << "\n";

    cout << "some Grad tests...\n";

    Col* x = new Col("x",1);
    InnerProduct* f = new InnerProduct(x, x);

    x->loadValues({3});
    f->compute(&cublasH);
    f->computeGrad(&cublasH);
    
    float* grad = new float;
    cudaMemcpy(grad, x->d_grad, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "I'm expecting that d/dx (x^2) at x=3 is 6, but I computed: " << *grad << ".\n";
    f->resetGrad();

    x->loadValues({9});
    f->compute(&cublasH);
    f->computeGrad(&cublasH);
    
    cudaMemcpy(grad, x->d_grad, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "I'm expecting that d/dx (x^2) at x=9 is 18, but I computed: " << *grad << ".\n";

    delete grad;
    
    cout << "Convolution tests...\n";
    Matrix* inputValues = new Matrix("input", 3, 3);
    Matrix* kernel = new Matrix("kernel", 2, 2);

    inputValues->loadValues({1,2,3,4,5,6,7,8,9});
    kernel->loadValues({3,3,3,3});


    Convolution* conv = new Convolution(inputValues, kernel, 0,1,0,1);

    conv->unrollKernel();

    float* testkernel = new float[conv->unrKrnlCols * conv->unrKrnlRows];
    cudaMemcpy(testkernel, conv->d_kernel, sizeof(float)*conv->unrKrnlCols*conv->unrKrnlRows, cudaMemcpyDeviceToHost);

    cout << "unrolled kernel looks like \n";
    outputMatrix(cout, testkernel, conv->unrKrnlRows, conv->unrKrnlCols);
    cout << "end unrolled kernel\n";


    conv->compute(&cublasH);

    float* testvalues = new float[4];
    cudaMemcpy(testvalues, conv->d_value, sizeof(float)*4, cudaMemcpyDeviceToHost);

    if(testvalues[0] != 36 || testvalues[1] != 48 || testvalues[2] != 72 || testvalues[3] != 84) {
        cout << "Convolution failed!  The output should be \n"
             << "( 36, 48 \n"
             << "  72, 84)\n"
             << "but it is\n";
        outputMatrix(cout, testvalues, 2, 2);
            
    }

    delete testvalues;
    delete testkernel;
    delete conv;

    inputValues = new Matrix("input", 4, 4);
    kernel = new Matrix("kernel", 3, 3);
    inputValues->loadValues({1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7});
    kernel->loadValues({1,0,0,0,1,0,0,0,1});


    conv = new Convolution(inputValues, kernel, 1, 3, 1, 3);

    conv->unrollKernel();

    testkernel = new float[conv->unrKrnlCols * conv->unrKrnlRows];
    cudaMemcpy(testkernel, conv->d_kernel, sizeof(float)*conv->unrKrnlCols*conv->unrKrnlRows, cudaMemcpyDeviceToHost);

    cout << "unrolled kernel looks like \n";
    outputMatrix(cout, testkernel, conv->unrKrnlRows, conv->unrKrnlCols);
    cout << "end unrolled kernel\n";

    conv->compute(&cublasH);
    testvalues = new float[4];
    cudaMemcpy(testvalues, conv->d_value, sizeof(float)*4, cudaMemcpyDeviceToHost);
    if(testvalues[0] != 7 || testvalues[1] != 4 || testvalues[2] != 4 || testvalues[3] != 9) {
        cout << "Convolution failed!  The output should be \n"
             << "( 7, 4 \n"
             << "  4, 9)\n"
             << "but it is\n";
        outputMatrix(cout, testvalues, 2, 2);
            
    }
    delete testvalues;
    delete testkernel;
    delete conv;






}
