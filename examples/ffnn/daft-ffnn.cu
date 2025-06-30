#include <cublas_v2.h>

#include "../../daft_autodiff/daft_autodiff.h"
#include "../../mnistdata/mnistdata.h"


using namespace std;
using namespace DA;
using namespace MNIST;

const unsigned int NUM_HIDDEN_NODES = 100;

int main() {
    cublasHandle_t cublasH;

    cublasCreate(&cublasH);
    Function *f;
    f = new Function(&cublasH);
    f->addOp(Operation::column("input", INPUT_SIZE + 1));
    f->addOp(Operation::matrix("weights1", NUM_HIDDEN_NODES, INPUT_SIZE + 1));
    f->addOp(Operation::matrix("weights2", OUTPUT_SIZE, NUM_HIDDEN_NODES));
    f->addOp(Operation::matrixProduct("layer1_output", "weights1", "input", NUM_HIDDEN_NODES, INPUT_SIZE + 1, 1));
    f->addOp(Operation::applyLeakyReLU("layer1_relu", "layer1_output", NUM_HIDDEN_NODES, 1));
    f->addOp(Operation::matrixProduct("prediction", "weights2", "layer1_relu", OUTPUT_SIZE, NUM_HIDDEN_NODES, 1));

    Training_Data testRows = load_data_from_file("../data/mnist_test.txt", 10000);



    



    
}
