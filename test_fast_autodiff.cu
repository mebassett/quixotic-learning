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
    
    float* test_value = new float;
    cudaMemcpy(test_value, test_ip->d_value, sizeof(float), cudaMemcpyDeviceToHost);

    if( *test_value != 11.0) {
        cout << "InnerProduct failed!  should be 11 but it is" << *test_value << ".\n";
    }

    cout << "Testing AddCol \n";

    AddCol* test_add = new AddCol(xy, ab);
    test_add->compute();

    float* test_values = new float[2];
    cudaMemcpy(test_values, test_add->d_value, 2*sizeof(float), cudaMemcpyDeviceToHost);

    if (test_values[0] != 4 || test_values[1] != 6 ) {
        cout << "AddCol failed! Should be  {4, 6} but its" 
             << "{" << test_values[0] << ", " << test_values[1] << "}\n";
    }

    cout << "Testing MatrixColProduct\n";

    Matrix* abcd = new Matrix("abcd", 2, 2);
    abcd->loadValues({1,-1,-1, 1});

    MatrixColProduct *test_matCol = new MatrixColProduct(abcd, xy);
    test_matCol->compute();

    cudaMemcpy(test_values, test_matCol->d_value, 2*sizeof(float), cudaMemcpyDeviceToHost);

    if (test_values[0] != -1 || test_values[1] != 1 ) {
        cout << "MatrixColProduct failed! Should be  {-1, 1} but its" 
             << "{" << test_values[0] << ", " << test_values[1] << "}\n";
    }




}
