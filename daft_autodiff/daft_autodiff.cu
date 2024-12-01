#include "daft_autodiff.h"
#include <algorithm>
#include <cublas_v2.h>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

namespace DA {
    //TODO move to cuda file shared between both silly and daft things.
    __global__ void doLeakyReLU(int Arows, int Acols, float* grad, float* A,
        float* result)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < Arows) && (col < Acols)) {
            int i = row * Acols + col;
            if (A[i] > 0) {
                grad[i] = 1;
                result[i] = A[i];
            } else {
                result[i] = 0.01 * A[i];
                grad[i] = 0.01;
            }
        }
    }
    __global__ void doFill(int rows, int cols, float value, float* result)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int i = row * cols + col;
        if (i < rows * cols)
            result[i] = value;
    }

    Operation Operation::column(string name, uint rows) {
        return { .opType=InputColumn
               , .workingSize = rows
               , .resultSize = 0
               , .gradSize = rows
               , .rows = rows
               , .cols = 1
               , .name{name} };
    }

    Operation Operation::multipleByMatrix(string name, uint rows, uint cols, string target) {
        return { .opType=MultiplyByMatrix
               , .workingSize = rows*cols
               , .resultSize = rows
               , .gradSize = rows*cols
               , .rows = rows
               , .cols = cols
               , .name{name} 
               , .config{ (BasicConfig) {.target{target}}}
               };
    }    

    Operation Operation::applyLeakyReLU(string name, string target) {
        BasicConfig config = { .target{target}};
        return { .opType=LeakyReLU
                , .workingSize = 0
                , .resultSize = 0
                , .gradSize = 0
                , .rows=0
                , .cols=0
                , .name{name}
                , .noOp = true 
                , .config{ (BasicConfig){ .target{target} } }
                };
    }

    Operation Operation::innerProduct(string name, string target1, string target2, uint rows) {
        return { .opType=InnerProduct
               , .workingSize = 0
               , .resultSize = 1
               , .gradSize = 2*rows
               , .rows = 1
               , .cols = 1
               , .name{name}
               , .noOp = false
               , .config{ (BinaryOpConfig) {.target1{target1}, .target2{target2}, 
                                            .targetRows = rows, .targetCols = 1}}
        };

    }

    Operation Operation::add(string name, string target1, string target2, uint rows, uint cols) {
        return { .opType=Add
               , .workingSize = 0
               , .resultSize = rows*cols
               , .gradSize = 2*rows*cols
               , .rows = rows
               , .cols = cols
               , .name{name}
               , .noOp = false
               , .config{ (BinaryOpConfig) {.target1{target1}, .target2{target2}, 
                                            .targetRows = rows, .targetCols = 1}}
        };
    }

    Operation Operation::scalarMultiply(string name, string target, uint rows, uint cols, float scale) {
        return { .opType=Scalar
               , .workingSize = 0
               , .resultSize = rows*cols
               , .gradSize = rows*cols
               , .rows = rows
               , .cols = cols
               , .name{name}
               , .noOp = false
               , .config{ (ScalarConfig) { .target{target}, .scale=scale }}
        };
    }

    Function::Function(cublasHandle_t* cublasH) : ops(), memLocs(), cublasH(cublasH) {
    }

    void Function::compile() {
        uint totalSize = 0;
        for(auto op : ops){
            totalSize += op.workingSize + op.resultSize + op.gradSize;
        }
        cudaMalloc((void**)&d_value,  totalSize  * sizeof(float));
        
        totalSize = 0;
        for(auto op:ops){
            memLocs[op.name+"_working"] = d_value + totalSize;
            totalSize += op.workingSize;
            memLocs[op.name+"_result"] = d_value + totalSize;
            totalSize += op.resultSize;
            memLocs[op.name+"_grad"] = d_value;
            totalSize += op.gradSize;

        }

    }
    Function::~Function() {
        cudaFree(d_value);
        
    }

    void Function::addOp(Operation op) {
        if(op.noOp) {
            Operation& lastOp = ops[ops.size()-1] ;
            Operation newOp { .opType = op.opType
                            , .workingSize = 0
                            , .resultSize = lastOp.resultSize
                            , .gradSize = lastOp.resultSize
                            , .rows = lastOp.rows
                            , .cols = lastOp.cols
                            , .name{op.name}
                            , .noOp = true 
                            , .config = lastOp.config
                            };
            ops.push_back(newOp);
        } else
            ops.push_back(op);
    }

    void Function::setValue(string name, vector<float> value) {
        // trusting the user to give us a vector of the right size! eek!    
        cudaMemcpy(memLocs[name+"_result"], &(value[0]), sizeof(float)*value.size(), cudaMemcpyHostToDevice);
    }

    void Function::getValue(string name, float* result) {
        Operation *op;
        for(auto needle : ops){
            if(needle.name == name) {
                op = &needle;
                break;
            }
        }
        float* d_value = memLocs[name+"_result"];
        cudaMemcpy(result, d_value, sizeof(float)*op->rows*op->cols,cudaMemcpyDeviceToHost);
    }

    void Function::getGrad(string name, float* result) {
        Operation *op;
        for(auto needle : ops){
            if(needle.name == name) {
                op = &needle;
                break;
            }
        }
        float* d_value = memLocs[name+"_grad"];
        cudaMemcpy(result, d_value, sizeof(float)*op->gradSize,cudaMemcpyDeviceToHost);
    }

    void Function::computeGrad(string name) {
        const auto op = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});
        if(op == ops.end()) return;
        float* seed;
        cudaError_t err;
        int size = op->rows * op->cols * sizeof(float);
        err = cudaMalloc((void**)&seed, size);
        if (err != cudaSuccess) {
            printf("malloc error in Function::computeGrad: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        dim3 gd(ceil(op->cols / 32.0), ceil(op->rows / 32.0), 1);
        dim3 bd(32, 32, 1);
        doFill<<<gd, bd>>>(op->rows, op->cols, 1.0f, seed);

        computeGrad(name, seed);
        

    }

    void Function::computeGrad(string name, float* seed){
        const auto it = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});

        if(it != ops.end()) {
            Operation op = *it;
            cout << "after find\n";
            if(op.opType == InnerProduct) {
                cout << "doing stuff with " << op.name << "\n";
                if(op.config.valueless_by_exception()){
                    cout << "valueless by exception\n";
                }
                BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                cout << "found " << op.name << "\n";
                float* col1 = memLocs[opConfig.target1+"_result"];
                float* col2 = memLocs[opConfig.target2+"_result"];

                float* vec1 = memLocs[name+"_grad"];
                float* vec2 = memLocs[name+"_grad"] + opConfig.targetRows;

                cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_DEVICE);
                cublasScopy(*cublasH, opConfig.targetRows, col1, 1, vec1, 1);
                cublasScopy(*cublasH, opConfig.targetRows, col2, 1, vec2, 1);



                cublasSscal(*cublasH, opConfig.targetRows, seed, vec1, 1);
                cublasSscal(*cublasH, opConfig.targetRows, seed, vec2, 1);
                cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_HOST );
                
                computeGrad(opConfig.target1, vec2);
                if(opConfig.target1 != opConfig.target2)
                    computeGrad(opConfig.target2, vec1);
                return;
            }
            if(op.opType == InputColumn) {
                float *grad = memLocs[name+"_grad"];

                float alpha = 1;
                cublasSaxpy(*cublasH, op.cols * op.rows, &alpha, seed, 1, grad, 1);
                
            }
        }
    }


    void Function::compute() {
        for(auto op : ops) {
            switch(op.opType) {
                case InputColumn:
                    // basically noop
                    break;
                break;
                case MultiplyByMatrix: {
                    BasicConfig opConfig = get<BasicConfig>(op.config);
                    float alpha = 1;
                    float beta = 0;
                    float* d_matrix = memLocs[op.name+"_working"];
                    float* d_col = memLocs[opConfig.target+"_result"];
                    float* d_result = memLocs[op.name+"_result"];

                    cublasSgemv(*cublasH, CUBLAS_OP_T, op.cols, op.rows,
                        &alpha, d_matrix, op.cols,
                        d_col, 1, &beta, d_result, 1);
                    

                break;}

                case LeakyReLU:{ 
                    BasicConfig opConfig = get<BasicConfig>(op.config);
                    float* d_col = memLocs[opConfig.target+"_result"];
                    float* d_grad = memLocs[op.name+"_grad"];
                    float* d_result = memLocs[op.name+"_result"];

                    dim3 bd(32, 32, 1);
                    dim3 gd(ceil(op.cols / 32.0), ceil(op.rows / 32), 1);

                    doLeakyReLU<<<gd, bd>>>(op.rows, op.cols, d_grad, d_col, d_result);


                break;}
                case Add:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = memLocs[opConfig.target1+"_result"];
                    float* d_v2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    float alpha = 1.0;
                    
                    cublasScopy(*cublasH, op.rows * op.cols, d_v1, 1, d_result, 1);
                    cublasSaxpy(*cublasH, op.rows * op.cols, &alpha, d_v2, 1, d_result, 1);
                break;}
                case Scalar:{
                    ScalarConfig opConfig = get<ScalarConfig>(op.config);
                    float* d_target = memLocs[opConfig.target+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    cublasScopy(*cublasH, op.rows * op.cols, d_target, 1, d_result, 1);
                    cublasSscal(*cublasH, op.rows * op.cols, &(opConfig.scale), d_result, 1);
                break;}
                case InnerProduct:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = memLocs[opConfig.target1+"_result"];
                    float* d_v2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    cublasSdot(*cublasH, opConfig.targetRows, d_v1, 1, d_v2, 1, d_result);

                break;}
            }
        }

    }

} // namespace DA
