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
    delete test_scalar;

    cout << "Testing InnerProduct \n";

    Col* ab = new Col("ab", 2);
    ab->loadValues({ 3.0, 4.0 });
    xy = new Col("xy", 2);
    xy->loadValues({ 1.0, 2.0});
    InnerProduct* test_ip = new InnerProduct(xy, ab);
    test_ip->compute(&cublasH);
    test_ip->fromDevice();
    

    if( *test_ip->value != 11.0) {
        cout << "InnerProduct failed!  should be 11 but it is" << *test_ip->value << ".\n";
    }
    delete test_ip;

    cout << "Testing Add \n";

    
    xy = new Col("xy", 2);
    xy->loadValues({ 1.0, 2.0});
    ab = new Col("ab", 2);
    ab->loadValues({3.0, 4.0});
    Add* test_add = new Add(xy, ab);
    test_add->compute(&cublasH);
    test_add->fromDevice();

    if (test_add->value[0] != 4 || test_add->value[1] != 6 ) {
        cout << "Add failed! Should be  {4, 6} but its" 
             << "{" << test_add->value[0] << ", " << test_add->value[1] << "}\n";
    }
    delete test_add;

    cout << "Testing MatrixColProduct\n";

    float *matrixGrad = new float[4];
    Matrix* abcd = new Matrix("abcd", 2, 2);
    abcd->loadValues({1,-1,-1, 1});
    xy = new Col("xy", 2);
    xy->loadValues({ 1.0, 2.0});
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
    delete test_matCol;

    cout << "Testing Leaky ReLU\n";

    Col* z = new Col("z", 4);
    ColLeakyReLU* relu = new ColLeakyReLU(z);
    z->loadValues({500, -500, 0.5, -1});
    relu->compute(&cublasH);
    relu->fromDevice();

    cout << relu->value[0] << ", " << relu->value[1] << ", " << relu->value[2] << ", " << relu->value[3]
         << "\n";
    delete relu;

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
    delete f;
    
    cout << "Convolution tests...\n";
    Matrix* inputValues = new Matrix("input", 3, 3);
    Matrix* kernel = new Matrix("kernel", 2, 2);

    inputValues->loadValues({1,2,3,4,5,6,7,8,9});
    kernel->loadValues({3,3,3,3});


    Convolution* conv = new Convolution(inputValues, kernel, 0,1,0,1);

    conv->unrollKernel();

    float* testkernel = new float[conv->unrKrnlCols * conv->unrKrnlRows];
    cudaMemcpy(testkernel, conv->d_kernel, sizeof(float)*conv->unrKrnlCols*conv->unrKrnlRows, cudaMemcpyDeviceToHost);



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
    cout << "PASSED!\n";

    cout << "Okay, now testing Convolution gradients...\n";

    inputValues = new Matrix("input", 2, 2);
    kernel = new Matrix("kernel", 2, 2);
    inputValues->loadValues({1,2,3,4});
    kernel->loadValues({3,3,3,3});
    conv = new Convolution(inputValues, kernel, 0, 1, 0, 1);

    conv->compute(&cublasH);
    conv->computeGrad(&cublasH);

    testvalues = new float[4];
    cudaMemcpy(testvalues, kernel->d_grad, sizeof(float)*4, cudaMemcpyDeviceToHost);
    if(testvalues[0] != 1 || testvalues[1] != 2 || testvalues[2] != 3 || testvalues[3] != 4) {
        cout << "Convolution gradient failed!  The kernel grad should be \n"
             << "( 1, 2\n"
             << "  3, 4)\nBut it is\n";
        outputMatrix(cout, testvalues, 2, 2);
    }
    delete testvalues;
    testvalues = new float[4];
    cudaMemcpy(testvalues, inputValues->d_grad, sizeof(float)*4, cudaMemcpyDeviceToHost);
    if(testvalues[0] != 3 || testvalues[1] != 3 || testvalues[2] != 3 || testvalues[3] != 3) {
        cout << "Convolution gradient failed!  The matrix grad should be \n"
             << "( 3, 3\n"
             << "  3, 3)\nBut it is\n";
        outputMatrix(cout, testvalues, 2, 2);
    }

    delete testvalues;
    delete conv;

    cout << "more grad tests...\n";

    Matrix* id3 = new Matrix("id3", 3, 3);
    id3->loadValues({1,0,0,0,1,0,0,0,1});

    Matrix* k2  = new Matrix("k2", 2, 2);
    k2->loadValues({0,1,1,0});

    Col* v = new Col("v", 2);
    v->loadValues({1,1});

    Convolution* c2 = new Convolution(id3, k2, 0, 1, 0, 1);
    MatrixColProduct* p = new MatrixColProduct(c2, v);
    InnerProduct* f1 = new InnerProduct(p,p);

    f1->compute(&cublasH);
    f1->computeGrad(&cublasH);

    float* testvalue = new float[1];
    cudaMemcpy(testvalue, f1->d_value, sizeof(float), cudaMemcpyDeviceToHost);
    if(*testvalue != 2) {
        cout << "expecting a value of 2 but it is " << *testvalue << "\n";
    }

    float* testgrad = new float[4];
    cudaMemcpy(testgrad, k2->d_grad, sizeof(float)*4, cudaMemcpyDeviceToHost);
    if(testgrad[0] != 4 || testgrad[1] != 2 || testgrad[2] != 2 || testgrad[3] != 4) {
        cout << "Convlution gradient failed! the kernel grad should be \n"
             << "( 4, 2\n"
             << "  2, 4\nbut it is\n";
        outputMatrix(cout, testgrad, 2, 2);
    }
    delete []testgrad;
    delete testvalue;
    delete f1;

    id3 = new Matrix("id3-2", 2, 2);
    id3->loadValues({0,1,-1,0});

    k2 = new Matrix("k2-2", 2,2);
    k2->loadValues({5,6,9,3});

    Matrix* k3 = new Matrix("k3", 2,2);
    k3->loadValues({1,1,1,1});

    c2 = new Convolution(id3, k2, 1, 2, 1, 2);
    Convolution* f2 = new Convolution(c2, k3, 0, 1, 0, 1);

    f2->compute(&cublasH);
    f2->computeGrad(&cublasH);

    testgrad = new float[4];
    cudaMemcpy(testgrad, k2->d_grad, sizeof(float)*4, cudaMemcpyDeviceToHost);
    if(testgrad[0] != 0 || testgrad[1] != -1 || testgrad[2] != 1 || testgrad[3] != 0) {
        cout << "Convlution gradient failed! the kernel grad should be \n"
             << "( 0, -1\n"
             << "  1, 0\nbut it is\n";
        outputMatrix(cout, testgrad, 2, 2);
    }
    delete [] testgrad;
    delete f2;

    cout << "PASSED!\n";

    cout << "Now testing MaxPool...\n";

    Matrix *m = new Matrix("m", 2, 2);
    m->loadValues({1,2,3,4});

    MaxPool *mp = new MaxPool(m, 2, 2, 1, 1);
    mp->compute(&cublasH);

    testvalue = new float[1];
    cudaMemcpy(testvalue, mp->d_value, sizeof(float), cudaMemcpyDeviceToHost);
    if(*testvalue != 4){
        cout << "MaxPool test failed.  expecting 4, the results was " << *testvalue << ".\n";
    }

    delete mp;
    delete testvalue;

    m = new Matrix("m-1", 4, 4);
    m->loadValues({1, 2, 1, 2,
                   3, 9, 16, 3,
                   1, 10,4, 1,
                   3, 4, 2, 3});
    
    mp = new MaxPool(m, 2, 2, 2, 2);
    mp->compute(&cublasH);

    testvalue = new float[4];
    cudaMemcpy(testvalue, mp->d_value, sizeof(float)*4, cudaMemcpyDeviceToHost);
    if(testvalue[0] != 9 || testvalue[1] != 16 || testvalue[2] != 10 || testvalue[3] != 4) {
        cout << "MaxPool test failed.  expected \n"
             << "( 9, 16\n"
             << "  10, 4)\nbut got\n";
        outputMatrix(cout, testvalue, 2, 2);
    }

    delete testvalue;
    delete mp;
    cout << "now testing gradients...\n";

    m = new Matrix("m-2", 2,2);
    m->loadValues({1,1,1,4});
    mp = new MaxPool(m, 2,2,1,1);
    
    Scalar* smp = new Scalar(mp, 5);
    smp->compute(&cublasH);
    smp->computeGrad(&cublasH);

    testvalue = new float[4];
    cudaMemcpy(testvalue, m->d_grad, sizeof(float)*4, cudaMemcpyDeviceToHost);
    if(testvalue[0] != 0 || testvalue[1] != 0 || testvalue[2] != 0 || testvalue[3] != 5) {
        cout << "MaxPool grad test failed.  expected \n"
             << "( 0, 0\n"
             << "  0, 5)\nbut got\n";
        outputMatrix(cout, testvalue, 2, 2);
    }
    delete [] testvalue;
    delete smp;

    cout << "now testing Flatten and its gradients...\n";

    m = new Matrix("m-3", 2, 2);
    m->loadValues({1,2,3,4});

    Flatten* flat = new Flatten(m);
    InnerProduct* flatF = new InnerProduct(flat, flat);
    flatF->compute(&cublasH);

    testvalue = new float[1];
    cudaMemcpy(testvalue, flatF->d_value, sizeof(float), cudaMemcpyDeviceToHost);
    if(*testvalue != 1.0 + 4 + 9 + 16) {
        cout << "Flatten failed.  expect 30 but got " << testvalue << "\n";
    }
    delete testvalue;
    delete flatF;

    m = new Matrix("m-4", 3, 3);
    m->loadValues({1,2,3,4,5,6,7,8,9});

    Matrix* k4 = new Matrix("k4", 2, 2);
    k4->loadValues({1,1,1,1});

    Col* v1 = new Col("v1", 4);
    v1->loadValues({1,2,3,4});

    Convolution* c1 = new Convolution(m, k4, 0, 1, 0, 1);
    flat = new Flatten(c1);
    flatF = new InnerProduct(flat, v1);
    flatF->compute(&cublasH);
    flatF->computeGrad(&cublasH);

    testvalue = new float[9];
    cudaMemcpy(testvalue, m->d_grad, 9*sizeof(float), cudaMemcpyDeviceToHost);
    if(testvalue[0] != 1 || testvalue[1] != 3 || testvalue[2] != 2
      || testvalue[3] != 4 || testvalue[4] != 10 || testvalue[5] != 6
      || testvalue[6] != 3 || testvalue[7] != 7 || testvalue[8] != 4) {
        cout << "Flatten grad test failed.  expected \n"
             << "( 1, 3 , 2 \n"
             << "  4, 10, 6\n"
             << "  3, 7 , 4)\nbut got\n";
        outputMatrix(cout, testvalue, 3, 3);
    }
    delete flatF;
    delete [] testvalue;

    cout << "now testing ConcatCol...\n";

    Col* v2 = new Col("v2", 1);
    v2->loadValues({1});
    Col* v3 = new Col("v3", 1);
    v3->loadValues({2});

    Col* v4 = new Col("v4", 2);
    v4->loadValues({3,5});

    AD* v2v3 = new ConcatCol({v2, v3});

    flatF = new InnerProduct(v2v3, v4);
    flatF->compute(&cublasH);

    testvalue = new float[1];
    cudaMemcpy(testvalue, flatF->d_value, sizeof(float), cudaMemcpyDeviceToHost);
    if(*testvalue != 13) {
        cout << "ConcatCol compute failed.  expecting 13 but got " << testvalue << ".\n";
    }

    flatF->computeGrad(&cublasH);

    testgrad = new float[1];
    cudaMemcpy(testgrad, v2->d_grad, sizeof(float), cudaMemcpyDeviceToHost);
    if(*testgrad != 3) {
        cout << "ConcatCol grad failed.  expecting 3 but got " << testvalue << ".\n";
    }
    delete flatF;
    delete [] testvalue;
    delete [] testgrad;

    cublasDestroy(cublasH);
}
