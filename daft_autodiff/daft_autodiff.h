// daft_autodiff.h
#ifndef DAFT_AUTODIFF_H
#define DAFT_AUTODIFF_H
#include <cublas_v2.h>
#include <map>
#include <string>
#include <vector>

using namespace std;

namespace DA {

enum OperationType { InputColumn, MultiplyByMatrix, LeakyReLU };

struct Operation {
    const OperationType opType;
    const uint workingSize;
    const uint resultSize;
    const uint gradSize;
    const uint rows;
    const uint cols;
    const string name;
    const bool noOp = false;

    static Operation column(string name, uint rows);
    static Operation multipleByMatrix(string name, uint rows, uint cols);
    static Operation applyLeakyReLU(string name);
};


struct Function {
    vector<Operation> ops;
    map<string, float*> memLocs;
    float* d_value;
    cublasHandle_t* cublasH;
    Function(cublasHandle_t* cublasH);
    void compile();
    void addOp(Operation); 
    void setValue(string name, vector<float> value);
    void compute();
};

}
#endif /* DAFT_AUTODIFF_H */
