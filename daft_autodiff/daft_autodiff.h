// daft_autodiff.h
#ifndef DAFT_AUTODIFF_H
#define DAFT_AUTODIFF_H
#include <cublas_v2.h>
#include <map>
#include <string>
#include <variant>
#include <vector>

using namespace std;

namespace DA {

enum OperationType { InputColumn, MultiplyByMatrix, LeakyReLU, Add, Scalar, InnerProduct};

struct BasicConfig {
    string target;
};

struct BinaryOpConfig {
    string target1;
    string target2;
    const uint targetRows;
    const uint targetCols;
};

struct ScalarConfig {
    string target;
    float scale;
};

using OpConfig = variant<BasicConfig, BinaryOpConfig, ScalarConfig>;

struct Operation {
    const OperationType opType;
    const uint workingSize;
    const uint resultSize;
    const uint gradSize;
    const uint rows;
    const uint cols;
    const string name;
    const bool noOp = false;
    const OpConfig config;

    static Operation column(string name, uint rows);
    static Operation multipleByMatrix(string name, uint rows, uint cols, string target);
    static Operation applyLeakyReLU(string name, string target);
    static Operation innerProduct(string name, string target1, string target2, uint rows);
    static Operation add(string name, string target1, string target2, uint rows, uint cols);
    static Operation scalarMultiply(string name, string target, uint rows, uint cols, float scale);
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
    void getValue(string name, float* result);
    void getGrad(string name, float* result);
    void computeGrad(string name, float* seed);
    void computeGrad(string name);
    ~Function();
};

}
#endif /* DAFT_AUTODIFF_H */
