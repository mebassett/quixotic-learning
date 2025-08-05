#include <cublas_v2.h>
#include <vector>
#include <random>

#include "../../daft_autodiff/daft_autodiff.h"
#include "../../mnistdata/mnistdata.h"


using namespace std;
using namespace DA;
using namespace MNIST;

int fromModelOutput(float* out) {
    float max = *max_element(out, out + OUTPUT_SIZE);
    for (int i { 0 }; i < OUTPUT_SIZE; i++) {
        if (*(out + i) == max)
            return i;
    }
    return -1;
}

int main() {
    // see https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf for architecture.
    cout << "Building LeNet...\n" << endl;

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    Function f = Function(&cublasH);

    f.addOp(Operation::matrix("featureInput", 28, 28));
    f.addOp(Operation::column("targetInput", 10));

    // its handy to add them in separate for-loops to keep the memory
    // contiguous.
    for (int i =0; i<6; i++) {
        f.addOp(Operation::matrix("c1-kernel-" + to_string(i), 5, 5));
    }
    for (int i =0; i<6; i++) {
        f.addOp(Operation::convolution("c1-conv-" + to_string(i)
                , "featureInput"
                , "c1-kernel"+to_string(i)
                , 2, 1, 2, 1
                , 28, 28
                , 5, 5));
    }
    //static Operation maxPool(string name, string target, uint width, uint height, 
    //        uint rowSkip, uint colSkip, uint targetRows, uint targetCols);
    //    int rows = (multiplicandRows + 2 * rowPadding - kernelRows) / rowSkip + 1;
    //    int cols = (multiplicandCols + 2 * colPadding - kernelCols) / colSkip + 1;
    //static Operation applyLeakyReLU(string name, string target, uint rows, uint cols);
    for (int i=0;i<6;i++) {
        f.addOp(Operation::applyLeakyReLU("c1-relu-conv-"+to_string(i)
                , "c1-conv-"+to_string(i)
                , 28, 28));
        
    }
    for (int i=0;i<6;i++) {
        f.addOp(Operation::maxPool("c1-pool-"+to_string(i)
                , "c1-relu-conv-"+to_string(i)
                , 2, 2, 2, 2
                , 28, 28));
    }



    cublasDestroy(cublasH);
    delete f;
}


