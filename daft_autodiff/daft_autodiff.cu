#include "daft_autodiff.h"
#include <algorithm>
#include <cublas_v2.h>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

namespace DA {


    ostream& operator<<(ostream &o, const OperationType t) {
        switch(t) {
            case OperationType::InputColumn:
                o << "InputColumn";
                break;
            case OperationType::InputMatrix:
                o << "InputMatrix";
                break;
            case OperationType::MatrixProduct:
                o << "MatrixProduct";
                break;
            case OperationType::LeakyReLU:
                o << "LeakyReLU";
                break;
            case OperationType::Add:
                o << "Add";
                break;
            case OperationType::Scalar:
                o << "Scalar";
                break;
            case OperationType::InnerProduct:
                o << "InnerProduct";
                break;

        }
        return o;
    }


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
        return { .opType=OperationType::InputColumn
               , .workingSize = 0
               , .resultSize = rows
               , .gradSize = rows
               , .rows = rows
               , .cols = 1
               , .name{name} };
    }

    Operation Operation::matrix(string name, uint rows, uint cols) {
        return { .opType=OperationType::InputMatrix
               , .workingSize = 0
               , .resultSize = rows*cols
               , .gradSize = rows*cols
               , .rows = rows
               , .cols = cols
               , .name{name} };
              
    }

    Operation Operation::matrixProduct(string name, string target1, string target2, uint target1Rows, uint target1Cols, uint target2Cols) {
        return { .opType=OperationType::MatrixProduct
               , .workingSize = 0 
               , .resultSize = target1Rows*target2Cols
               , .gradSize = target1Rows*target1Cols + target1Cols*target2Cols
               , .rows = target1Rows
               , .cols = target2Cols
               , .name{name} 
               , .config{ (BinaryMatrixConfig) { .target1{target1}
                                               , .target2{target2}
                                               , .target1Rows = target1Rows
                                               , .target1Cols = target1Cols
                                               , .target2Cols = target2Cols
                                               }}
               };
    }    

    Operation Operation::applyLeakyReLU(string name, string target) {
        BasicConfig config = { .target{target}};
        return { .opType=OperationType::LeakyReLU
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
        return { .opType=OperationType::InnerProduct
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
        return { .opType=OperationType::Add
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
        return { .opType=OperationType::Scalar
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
        int totalSize = 0;
        for(auto op : ops){
            totalSize += op.workingSize + op.resultSize + op.gradSize;
        }
        cudaMalloc((void**)&d_value,  totalSize  * sizeof(float));
        cudaMemset(d_value, 0, totalSize * sizeof(float));
        dim3 gd(1, ceil(totalSize / 32.0), 1);
        dim3 bd(1, 1024, 1);
        doFill<<<gd, bd>>>(totalSize, 1, 0.0f, d_value);
        cudaDeviceSynchronize();
        
        totalSize = 0;
        for(auto op:ops){
            memLocs[op.name+"_working"] = d_value + totalSize;
            memLocs[op.name+"_result"] = d_value + totalSize + op.workingSize;
            memLocs[op.name+"_grad"] = d_value + totalSize + op.workingSize + op.resultSize;
            totalSize += op.gradSize + op.resultSize + op.workingSize;

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
        cudaMemcpy(memLocs[name+"_working"], &(value[0]), sizeof(float)*value.size(), cudaMemcpyHostToDevice);
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
        cudaDeviceSynchronize();

        computeGrad(name, seed);
        cudaDeviceSynchronize();
        cudaFree(seed);
        

    }

    void Function::computeGrad(string name, float* seed){
        const auto it = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});

        if(it != ops.end()) {
            Operation op = *it;
            float *grad = memLocs[name+"_grad"];
            switch(op.opType) {
                case OperationType::InnerProduct: {
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* col1 = memLocs[opConfig.target1+"_result"];
                    float* col2 = memLocs[opConfig.target2+"_result"];


                    float* vec1 = memLocs[name+"_grad"];
                    float* vec2 = memLocs[name+"_grad"] + opConfig.targetRows;
                    
                    cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_DEVICE);

                    cublasScopy(*cublasH, opConfig.targetRows, col1, 1, vec1, 1);
                    cublasSscal(*cublasH, opConfig.targetRows, seed, vec1, 1);
                    
                    cublasScopy(*cublasH, opConfig.targetRows, col2, 1, vec2, 1);
                    cublasSscal(*cublasH, opConfig.targetRows, seed, vec2, 1);
                    cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_HOST );

                    
                    computeGrad(opConfig.target2, vec1);
                    computeGrad(opConfig.target1, vec2);
                } break;
                case OperationType::InputColumn: {
                    float *grad = memLocs[name+"_grad"];
                    float alpha = 1;
                    cublasSaxpy(*cublasH, op.gradSize, &alpha, seed, 1, grad, 1);
                } break;
                case OperationType::InputMatrix: {
                    float *grad = memLocs[name+"_grad"];
                    float alpha = 1;
                    cublasSaxpy(*cublasH, op.gradSize, &alpha, seed, 1, grad, 1);
                } break;
                case OperationType::MatrixProduct: {
                    BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

                    float *matrixValue = memLocs[opConfig.target1+"_result"];
                    float *colValue = memLocs[opConfig.target2+"_result"];
                    float *matrixGrad = memLocs[op.name+"_grad"];
                    float *colGrad = matrixGrad + opConfig.target1Rows * opConfig.target1Cols;

                    float alpha = 1;
                    float beta = 0;

                    cublasSgemm( *cublasH
                               , CUBLAS_OP_T
                               , CUBLAS_OP_N
                               , opConfig.target1Cols
                               , opConfig.target1Rows
                               , 1
                               , &alpha
                               , colValue
                               , 1
                               , seed
                               , 1
                               , &beta
                               , matrixGrad
                               , opConfig.target1Cols); 
                    cublasSgemm( *cublasH
                               , CUBLAS_OP_N
                               , CUBLAS_OP_T
                               , 1
                               , opConfig.target1Cols
                               , opConfig.target1Rows
                               , &alpha
                               , seed
                               , 1
                               , matrixValue
                               , opConfig.target1Cols
                               , &beta
                               , colGrad
                               , 1);
                    computeGrad(opConfig.target1, matrixGrad);
                    computeGrad(opConfig.target2, colGrad);
                } break;
                case OperationType::Scalar: {
                    ScalarConfig opConfig = get<ScalarConfig>(op.config);
                    float *grad = memLocs[op.name+"_grad"];
                    float *result = memLocs[op.name+"_result"];

                    cublasScopy(*cublasH, op.cols * op.rows, seed, 1, grad, 1);
                    cublasSscal(*cublasH, op.cols * op.rows, &(opConfig.scale), grad, 1);
                    computeGrad(opConfig.target, grad);

                } break;
                case OperationType::Add: {
                } break;
                case OperationType::LeakyReLU: {
                } break;

            }
        }
    }


    void Function::compute() {
        for(auto op : ops) {
            switch(op.opType) {
                case OperationType::InputColumn:
                    // basically noop
                break;
                case OperationType::InputMatrix:
                    // this is the same as InputColumn.  InputColumn is redundant
                break;
                case OperationType::MatrixProduct: {
                    BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);
                    float alpha = 1;
                    float beta = 0;
                    float* d_matrix1 = memLocs[opConfig.target1+"_result"];
                    float* d_matrix2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];

                    if(CUBLAS_STATUS_SUCCESS != cublasSgemm( *cublasH
                               , CUBLAS_OP_N
                               , CUBLAS_OP_N
                               , opConfig.target1Rows
                               , opConfig.target2Cols
                               , opConfig.target1Cols
                               , &alpha
                               , d_matrix1
                               , opConfig.target1Rows
                               , d_matrix2
                               , opConfig.target1Cols
                               , &beta
                               , d_result
                               , op.rows)) {
                        cout << "failing due to error" << endl;
                        exit(-1);
                    }

                break;}

                case OperationType::LeakyReLU:{ 
                    BasicConfig opConfig = get<BasicConfig>(op.config);
                    float* d_col = memLocs[opConfig.target+"_result"];
                    float* d_grad = memLocs[op.name+"_grad"];
                    float* d_result = memLocs[op.name+"_result"];

                    dim3 bd(32, 32, 1);
                    dim3 gd(ceil(op.cols / 32.0), ceil(op.rows / 32), 1);

                    doLeakyReLU<<<gd, bd>>>(op.rows, op.cols, d_grad, d_col, d_result);


                break;}
                case OperationType::Add:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = memLocs[opConfig.target1+"_result"];
                    float* d_v2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    float alpha = 1.0;
                    
                    cublasScopy(*cublasH, op.rows * op.cols, d_v1, 1, d_result, 1);
                    cublasSaxpy(*cublasH, op.rows * op.cols, &alpha, d_v2, 1, d_result, 1);
                break;}
                case OperationType::Scalar:{
                    ScalarConfig opConfig = get<ScalarConfig>(op.config);
                    float* d_target = memLocs[opConfig.target+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    cublasScopy(*cublasH, op.rows * op.cols, d_target, 1, d_result, 1);
                    cublasSscal(*cublasH, op.rows * op.cols, &(opConfig.scale), d_result, 1);
                break;}
                case OperationType::InnerProduct:{
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
