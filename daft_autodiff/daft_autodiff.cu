#include "daft_autodiff.h"
#include <cublas_v2.h>
#include <cassert>
#include <vector>

using namespace std;

namespace DA {

    Operation Operation::column(string name, uint rows) {
        return { .opType=InputColumn
               , .workingSize = rows
               , .resultSize = 0
               , .gradSize = rows
               , .rows = rows
               , .cols = 1
               , .name{name} };
    }

    Operation Operation::multipleByMatrix(string name, uint rows, uint cols) {
        return { .opType=MultiplyByMatrix
               , .workingSize = rows*cols
               , .resultSize = rows
               , .gradSize = rows*cols
               , .rows = rows
               , .cols = cols
               , .name{name} };
    }    

    Operation Operation::applyLeakyReLU(string name) {
        return { .opType=LeakyReLU
                , .workingSize = 0
                , .resultSize = 0
                , .gradSize = 0
                , .rows=0
                , .cols=0
                , .name{name}
                , .noOp = true 
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
                            , .noOp = true };
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
                break;
                case MultiplyByMatrix: {
                    float alpha = 1;
                    float beta = 0;
                    float* d_matrix = memLocs[op.name+"_working"];
                    float* d_col = memLocs[lastOp.name+"_result"];
                    float* d_result = memLocs[op.name+"_result"];

                    cublasSgemv(*cublasH, CUBLAS_OP_T, op.cols, op.rows,
                        &alpha, d_matrix, op.cols,
                        d_col, 1, &beta, d_result, 1);
                    

                break;}

                case LeakyReLU:{ 

                break;}
            }
        }

    }

} // namespace DA
