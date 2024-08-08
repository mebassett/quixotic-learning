#include <iostream>
#include <valarray>
#include "fast_autodiff.h"

using namespace std;
using namespace FA;

int main() {
    cout << "Testing Scalar \n";

    Col* xy = new Col("xy", 2);

    xy->loadValues({ 1.0, 2.0});
    
    Scalar* test_scalar = new Scalar(xy, 5);
    test_scalar->compute();
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
    test_ip->compute();
    test_ip->fromDevice();
    

    if( *test_ip->value != 11.0) {
        cout << "InnerProduct failed!  should be 11 but it is" << *test_ip->value << ".\n";
    }

    cout << "Testing AddCol \n";

    AddCol* test_add = new AddCol(xy, ab);
    test_add->compute();
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
    test_matCol->compute();
    test_matCol->fromDevice();
    test_matCol->computeGrad();

    cudaMemcpy(matrixGrad, abcd->d_grad, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Matrix grad\n"
         << matrixGrad[0] << ", " << matrixGrad[1] << "\n" 
         << matrixGrad[2] << ", " << matrixGrad[3] << "\n";

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
    relu->compute();
    relu->fromDevice();

    cout << relu->value[0] << ", " << relu->value[1] << ", " << relu->value[2] << ", " << relu->value[3]
         << "\n";




}
