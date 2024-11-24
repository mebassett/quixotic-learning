#include "daft_autodiff.h"
#include <cublas_v2.h>
#include <cassert>
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
               , .config{ (BinaryOpConfig) {.target1{target1}, .target2{target2}}}
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
               , .config{ (BinaryOpConfig) { .target1{target1}, .target2{target2}}}
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

    void Function::addOp(Operation op) {
        assert(("First op must be real!", !op.noOp > 0 && ops.size() > 0));
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

    void Function::compute() {
        for(uint i = 1; i<ops.size(); i++) {
            Operation op = ops[i];
            Operation lastOp = ops[i-1];
            switch(op.opType) {
                case InputColumn:
                    // basically noop
                    continue;
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
            }
        }

    }

} // namespace DA
