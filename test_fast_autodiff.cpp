#include <iostream>
#include<valarray>
import fast_autodiff;

using namespace std;

int main () {
    cout << "Tests for fast autodiff\n";
    cout << "Testing Inner Product\n";

    Col xy ("xy", 2);
    Col ab ("ab", 2);

    xy.loadValues({ 1.0, 2.0});
    ab.loadValues({ 3.0, 4.5});

    double seed[] = {1};

    InnerProduct test_innerProduct (&ab, &xy);

    test_innerProduct.compute();
    test_innerProduct.pushGrad(seed);

    if(*test_innerProduct.value != 12.0){
        cout << "Expecting inner product of <1,2> and <3, 4.5> to be 12, but it was " << *test_innerProduct.value << ".\n";
        return 1;
    }
    cout << "inner product compute passed.\n";

    if(!(*xy.grad == *ab.value && *(xy.grad+1) == *(ab.value+1))) {
        cout << "expecting grad of x to be 3, but it is " << *xy.grad << ".\n";
        cout << "expecting grad of y to be 4.5, but it is " << *(xy.grad+1) << ".\n";
        return 1;
    }
    cout << "inner product grad passed.\n";

    Matrix abcd ("abcd", 2, 2);
    abcd.loadValues({11, 15, 34, 2});

    MatrixColProduct test_matCol (&abcd, &xy);
    double seed1[2] = {1,1};

    test_matCol.resetGrad();
    test_matCol.compute();
    test_matCol.pushGrad(seed1);

    cout << *(abcd.grad+2) << "\n";
    cout << *(abcd.grad+2) << "\n";


}
