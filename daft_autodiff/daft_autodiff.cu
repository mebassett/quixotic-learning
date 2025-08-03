#include "daft_autodiff.h"
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

namespace DA {

#define cudaErrCk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line) {
    if(err != cudaSuccess) {
        fprintf(stderr, "CudaAssert: %s %s %d\n", cudaGetErrorString(err), file, line);
        exit(err);
    }
}

#define cublasErrCk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t err, const char *file, int line) {
    if(err != CUBLAS_STATUS_SUCCESS) {
      const char* errMsg;
      switch(err) {
         case CUBLAS_STATUS_NOT_INITIALIZED: errMsg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
         case CUBLAS_STATUS_ALLOC_FAILED: errMsg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
         case CUBLAS_STATUS_INVALID_VALUE: errMsg = "CUBLAS_STATUS_INVALID_VALUE"; break;
         case CUBLAS_STATUS_ARCH_MISMATCH: errMsg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
         case CUBLAS_STATUS_MAPPING_ERROR: errMsg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
         case CUBLAS_STATUS_EXECUTION_FAILED: errMsg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
         case CUBLAS_STATUS_INTERNAL_ERROR: errMsg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
         case CUBLAS_STATUS_NOT_SUPPORTED: errMsg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
         case CUBLAS_STATUS_LICENSE_ERROR: errMsg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
         default: errMsg = "Unknown CUBLAS error"; break;
      }
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", errMsg, file, line); 
      exit(err);
    }
}

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
            case OperationType::Convolution:
                o << "Convolution";
                break;
            case OperationType::MaxPool:
                o << "MaxPool";
                break;
            case OperationType::Concat:
                o << "Concat";
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

    __global__ void doComponentProduct(int rows, int cols, float* grad, float* seed,
        float* result)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows && col < cols) {
            result[row * cols + col] = seed[row * cols + col] * grad[row * cols + col];
        }
    }

    __global__ void doPadInput(float* input, float* paddedInput, int inputRows,
        int inputCols, int rowPadding, int colPadding)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int rows = inputRows + 2 * rowPadding;
        int cols = inputCols + 2 * colPadding;
    
        if (row < rows && col < cols) {
            if (row - rowPadding >= 0 && row < inputRows + rowPadding && col - colPadding >= 0 && col < inputCols + colPadding)
                paddedInput[row * cols + col] = input[(row - rowPadding) * inputCols + col - colPadding];
            else
                paddedInput[row * cols + col] = 0;
        }
    }

    __global__ void doUnroll(float* kernel, float* matrix, int kernelRows,
        int kernelCols, int mRows, int mCols, int inCols,
        int outCols, int rowSkip, int colSkip)
    {
        int mrow = blockIdx.y * blockDim.y + threadIdx.y;
        int mcol = blockIdx.x * blockDim.x + threadIdx.x;
        if (mrow < mRows && mcol < mCols) {
            int outRow = mrow / outCols;
            int outCol = mrow % outCols;
    
            int inRow = mcol / inCols;
            int inCol = mcol % inCols;
    
            int kRowIndex = inRow - rowSkip * outRow;
            int kColIndex = inCol - colSkip * outCol;
    
            if (kRowIndex >= 0 && kRowIndex < kernelRows && kColIndex >= 0 && kColIndex < kernelCols) {
                matrix[mrow * mCols + mcol] = kernel[kRowIndex * kernelCols + kColIndex];
            } else {
                matrix[mrow * mCols + mcol] = 0;
            }
        }
    }

    __global__ void doKernelRoll(float* matrix, float* kernel, int kernelRows,
        int kernelCols, int mCols, int mRows, int colSkip,
        int rowSkip, int inCols, int outCols)
    {
        int mcol = blockIdx.x * blockDim.x + threadIdx.x;
        if (mcol < mCols) {
            int kRow = mcol / inCols;
            int kCol = mcol % inCols;
            if (kRow >= 0 && kRow < kernelRows && kCol >= 0 && kCol < kernelCols) {
                float val = 0;
                for (int mrow = 0; mrow < mRows; mrow++) {
                    int ocol = mrow % outCols;
                    int orow = mrow / outCols;
    
                    int offset = colSkip * ocol + rowSkip * orow * inCols;
    
                    val += matrix[mrow * mCols + mcol + offset];
                }
                kernel[kRow * kernelCols + kCol] = val;
            }
        }
    }

    __global__ void doCopyWithoutPadding(float* paddedSource, float* dest, int rows,
        int cols, int rowPadding, int colPadding)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows && col < cols) {
            dest[row * cols + col] = paddedSource[(row + rowPadding) * (cols + 2 * colPadding) + col + colPadding];
        }
    }

    __global__ void doMaxPool(int targetRows, int targetCols, int matrixRows,
        int matrixCols, int rowSkip, int height, int colSkip,
        int width, float* matrix, float* result)
    {
        int trow = blockIdx.y * blockDim.y + threadIdx.y;
        int tcol = blockIdx.x * blockDim.x + threadIdx.x;
        if (trow < targetRows && tcol < targetCols) {
            float val = -9999;
            int mrow = rowSkip * trow;
            int mcol = colSkip * tcol;

            for (int i = mrow; i < mrow + height; i++)
                for (int j = mcol; j < mcol + width; j++)
                    if (i >= 0 && j >= 0 && i < matrixRows && j < matrixCols) {
                        int mIndex = matrixCols * i + j;
                        if (matrix[mIndex] > val)
                            val = matrix[mIndex];
                    }

            result[targetCols * trow + tcol] = val;
        }
    }

    __global__ void doMaxPoolGrad(int targetRows, int targetCols, int matrixRows,
        int matrixCols, int rowSkip, int height,
        int colSkip, int width, float* matrix,
        float* value, float* seed, float* result)
    {
        int trow = blockIdx.y * blockDim.y + threadIdx.y;
        int tcol = blockIdx.x * blockDim.x + threadIdx.x;
        int tIndex = targetCols * trow + tcol;
        if (trow < targetRows && tcol < targetCols) {
            int mrow = rowSkip * trow;
            int mcol = colSkip * tcol;

            float val = seed[tIndex];

            for (int i = mrow; i < mrow + height; i++)
                for (int j = mcol; j < mcol + width; j++)
                    if (i >= 0 && j >= 0 && i < matrixRows && j < matrixCols) {
                        int mIndex = matrixCols * i + j;
                        result[mIndex] = val * (matrix[mIndex] == value[tIndex]);
                    }
        }
    }

    void convolutionPadInput( uint inputRows, uint inputCols
                            , uint rowPadding, uint colPadding
                            , float* input, float* output) {
        uint outputRows = inputRows + rowPadding*2;
        uint outputCols = inputCols + colPadding*2;
        dim3 gd(ceil(outputCols / 32.0), ceil(outputRows / 32.0), 1);
        dim3 bd(32, 32, 1);
        doPadInput<<<gd, bd>>>(input, output, inputRows
                    , inputCols, rowPadding, colPadding);
        cudaErrCk( cudaPeekAtLastError() );
        cudaDeviceSynchronize(); //do you actually need this? 
    }

    void convolutionUnrollKernel( uint unrKrnlRows, uint unrKrnlCols
                                , uint kernelRows, uint kernelCols
                                , uint paddedInputCols
                                , uint outputCols
                                , uint rowSkip
                                , uint colSkip
                                , float* kernel
                                , float* output ) {
        dim3 gd(ceil(unrKrnlCols / 32.0), ceil(unrKrnlRows / 32.0), 1);
        dim3 bd(32, 32, 1);
        doUnroll<<<gd, bd>>>( kernel
                            , output
                            , kernelRows
                            , kernelCols
                            , unrKrnlRows
                            , unrKrnlCols
                            , paddedInputCols
                            , outputCols
                            , rowSkip
                            , colSkip );
        cudaErrCk( cudaPeekAtLastError() );
        cudaDeviceSynchronize(); //do you actually need this? 
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

    Operation Operation::applyLeakyReLU(string name, string target, uint rows, uint cols) {
        BasicConfig config = { .target{target}};
        return { .opType=OperationType::LeakyReLU
                , .workingSize = rows*cols
                , .resultSize = rows*cols
                , .gradSize = rows*cols
                , .rows=rows
                , .cols=cols
                , .name{name}
                , .noOp = false 
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
               , .gradSize = rows*cols
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
    
    Operation Operation::convolution(string name, string multiplicand, string kernel, uint rowPadding,
            uint rowSkip, uint colPadding, uint colSkip, 
            uint multiplicandRows, uint multiplicandCols,
            uint kernelRows, uint kernelCols) {
        int rows = (multiplicandRows + 2 * rowPadding - kernelRows) / rowSkip + 1;
        int cols = (multiplicandCols + 2 * colPadding - kernelCols) / colSkip + 1;

        int unrKrnlCols = (multiplicandRows + rowPadding*2) * (multiplicandCols + colPadding*2);
        int unrKrnlRows = rows * cols; 

        // working size ihe padding input plus the unrolled kernel.
        int paddedInputSize = 
          (multiplicandRows + 2 * rowPadding) * (multiplicandCols + 2 * colPadding);

        // grad size will be 
        //    size of the original kernel matrix
        //  + size of the original input matrix (multiplicand)
        //  BUT we also need some working memory for the unrolling and unpadding
        //  so we also have
        //  + size of unrolled kernel matrix
        //  + size of a column of the unrolled kernel matrix
        // of the multiplicand plus
        int gradSize =    (kernelRows * kernelCols) 
                        + (multiplicandRows * multiplicandCols)
                        + (unrKrnlRows * unrKrnlCols)
                        + unrKrnlCols;
        
        return { .opType=OperationType::Convolution
               , .workingSize = paddedInputSize + (unrKrnlRows * unrKrnlCols)
               , .resultSize = rows * cols
               , .gradSize = gradSize
               , .rows = rows
               , .cols = cols
               , .name{name}
               , .noOp = false
               , .config{ (ConvolutionConfig) {
                   .multiplicand{multiplicand}
                 , .kernel{kernel}
                 , .rowPadding = rowPadding
                 , .rowSkip = rowSkip
                 , .colPadding = colPadding
                 , .colSkip = colSkip
                 , .multiplicandRows = multiplicandRows
                 , .multiplicandCols = multiplicandCols
                 , .kernelRows = kernelRows
                 , .kernelCols = kernelCols
                 , .unrKrnlRows = unrKrnlRows
                 , .unrKrnlCols = unrKrnlCols
                 , .paddedInputSize = paddedInputSize
                 }}
               };
    }

    Operation Operation::maxPool(string name, string target, uint width, uint height, 
            uint rowSkip, uint colSkip, uint targetRows, uint targetCols) {
        uint rows = (targetRows - height) / rowSkip + 1;
        uint cols = (targetCols - width) / colSkip + 1;
        
        return { .opType=OperationType::MaxPool
               , .workingSize = 0
               , .resultSize = rows * cols
               , .gradSize = targetRows * targetCols
               , .rows = rows
               , .cols = cols
               , .name{name}
               , .noOp = false
               , .config{ (MaxPoolConfig) {
                   .target{target}
                 , .width = width
                 , .height = height
                 , .rowSkip = rowSkip
                 , .colSkip = colSkip
                 , .targetRows = targetRows
                 , .targetCols = targetCols
                 }}
               };
    }

    // CAUTION - this does a copy within device memory.
    // you can probably avoid this entirely by ensuring that all your results
    // end up in one continuous block and then lying about the size of the 
    // first target as an input into something else.
    Operation Operation::concat(string name, const vector<string>& targets, uint size) {
        return { .opType=OperationType::Concat
               , .workingSize = 0
               , .resultSize = size
               , .gradSize = 0
               , .rows = size
               , .cols = 1
               , .name{name}
               , .noOp = false
               , .config { (ConcatConfig) {
                     .targets{targets}
                   , .size=size
               }}
        };
    }

    Function::Function(cublasHandle_t* cublasH) : ops(), memLocs(), cublasH(cublasH) {
    }

    void Function::compile() {
        totalSize = 0; gradSize = 0; workingSize = 0; resultSize = 0;
        for(auto op : ops){
            totalSize += op.workingSize + op.resultSize + op.gradSize;
            gradSize += op.gradSize;
            workingSize += op.workingSize;
            resultSize += op.resultSize;
        }
        cudaMalloc((void**)&d_value,  totalSize  * sizeof(float));
        cudaMemset(d_value, 0, totalSize * sizeof(float));
        
        int idxGradSize = 0, idxWorkingSize = 0, idxResultSize = 0;
        for(auto op:ops){
            memLocs[op.name+"_grad"] = d_value + idxGradSize;
            idxGradSize += op.gradSize;
            memLocs[op.name+"_result"] = d_value + gradSize + idxResultSize;
            idxResultSize += op.resultSize;
            memLocs[op.name+"_working"] = d_value + gradSize + resultSize + idxWorkingSize;
            idxWorkingSize += op.workingSize;

        }
    }

    void Function::resetGrad() {
        cudaErrCk( cudaMemset(d_value, 0, gradSize * sizeof(float)) );
        cudaErrCk( cudaMemset(d_value+gradSize+resultSize, 0, workingSize * sizeof(float)) );
        
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

    void Function::gradDescent(string name, float learningRate) {
        float alpha = 1;
        float beta = -1 * learningRate;
        Operation *op;

        for(auto needle : ops){
            if(needle.name == name) {
                op = &needle;
                break;
            }
        }
        if(op->opType != OperationType::InputMatrix) return;

        float* result  = memLocs[name+"_result"];
        float* grad  = memLocs[name+"_grad"];


        cublasErrCk ( cublasSgeam( *cublasH
                     , CUBLAS_OP_N
                     , CUBLAS_OP_N
                     , op->cols, op->rows, &alpha
                     , result, op->cols, &beta
                     , grad, op->cols, result, op->cols) );

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
        if(op == ops.end()) {
          cout << "computeGrad cannot find op " << name << endl;
          exit(1);
          return;
        }
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
        cudaFree(seed);
        

    }

    void Function::computeGrad(string name, float* seed){
        const auto it = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});
        if(it == ops.end()) {
          cout << "computeGrad* cannot find op " << name << endl;
          exit(1);
          return;
        }

        Operation op = *it;
        float *grad = memLocs[name+"_grad"];
        switch(op.opType) {
            case OperationType::InnerProduct: {
                BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                float* col1 = memLocs[opConfig.target1+"_result"];
                float* col2 = memLocs[opConfig.target2+"_result"];

                float* vec1 = memLocs[name+"_grad"];
                float* vec2 = memLocs[name+"_grad"] + opConfig.targetRows;
                
                cublasErrCk( cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_DEVICE) );

                cublasErrCk( cublasScopy(*cublasH, opConfig.targetRows, col1, 1, vec1, 1) );
                cublasErrCk( cublasSscal(*cublasH, opConfig.targetRows, seed, vec1, 1) );
                
                cublasErrCk( cublasScopy(*cublasH, opConfig.targetRows, col2, 1, vec2, 1) );
                cublasErrCk( cublasSscal(*cublasH, opConfig.targetRows, seed, vec2, 1) );
                cublasErrCk( cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_HOST ) );

                
                computeGrad(opConfig.target2, vec1);
                computeGrad(opConfig.target1, vec2);
            } break;
            case OperationType::InputColumn: {
                float alpha = 1;
                cublasErrCk( cublasSaxpy(*cublasH, op.gradSize, &alpha, seed, 1, grad, 1) );
            } break;
            case OperationType::InputMatrix: {
                float alpha = 1;
                cublasErrCk( cublasSaxpy(*cublasH, op.gradSize, &alpha, seed, 1, grad, 1) );
            } break;
            case OperationType::MatrixProduct: {
                BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

                float *matrixValue = memLocs[opConfig.target1+"_result"];
                float *colValue = memLocs[opConfig.target2+"_result"];
                float *matrixGrad = memLocs[op.name+"_grad"];
                float *colGrad = matrixGrad + opConfig.target1Rows * opConfig.target1Cols;

                float alpha = 1;
                float beta = 0;

                cublasErrCk( cublasSgemm( *cublasH
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
                           , opConfig.target1Cols) ); 
                cublasErrCk( cublasSgemm( *cublasH
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
                           , 1) );
                computeGrad(opConfig.target1, matrixGrad);
                computeGrad(opConfig.target2, colGrad);
            } break;
            case OperationType::Scalar: {
                ScalarConfig opConfig = get<ScalarConfig>(op.config);
                float *result = memLocs[op.name+"_result"];

                cublasErrCk( cublasScopy(*cublasH, op.cols * op.rows, seed, 1, grad, 1) );
                cublasErrCk( cublasSscal(*cublasH, op.cols * op.rows, &(opConfig.scale), grad, 1) );
                computeGrad(opConfig.target, grad);

            } break;
            case OperationType::Add: {
                BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                float* copySeed = memLocs[op.name+"_grad"];

                cudaMemcpy(copySeed, seed, op.rows * op.cols * sizeof(float), cudaMemcpyDeviceToDevice);
                computeGrad(opConfig.target1, seed);
                computeGrad(opConfig.target2, copySeed);
            } break;
            case OperationType::LeakyReLU: {
                BasicConfig opConfig = get<BasicConfig>(op.config);
                float* newSeed = memLocs[op.name+"_working"];

                dim3 bd(32, 32, 1);
                dim3 gd(ceil(op.cols / 32.0), ceil(op.cols / 32.0), 1);

                doComponentProduct<<<gd, bd>>>(op.rows, op.cols, grad, seed, newSeed);

                computeGrad(opConfig.target, newSeed);

            } break;
            case OperationType::Convolution: {
                ConvolutionConfig opCnfg = get<ConvolutionConfig>(op.config);

                float alpha = 1;
                float beta = 0;

                int startInput = opCnfg.kernelRows * opCnfg.kernelCols;
                int startMatrix = startInput 
                    + (opCnfg.multiplicandRows * opCnfg.multiplicandCols);
                int startCol = startMatrix
                    + (opCnfg.unrKrnlRows * opCnfg.unrKrnlCols);

                float* rolledKernelMatrixGrad = memLocs[op.name+"_grad"];
                float* inputGrad = memLocs[op.name+"_grad"] + startInput;
                float* matrixGrad = memLocs[op.name+"_grad"] + startMatrix; 
                float* colGrad = memLocs[op.name+"_grad"] + startCol; 

                float* paddedInput = memLocs[op.name+"_working"];
                float* unrolledKernel = memLocs[op.name+"_working"]
                                    + opCnfg.paddedInputSize;

                cublasErrCk( cublasSgemm( *cublasH
                            , CUBLAS_OP_T
                            , CUBLAS_OP_N
                            , opCnfg.unrKrnlCols
                            , opCnfg.unrKrnlRows
                            , 1
                            , &alpha
                            , paddedInput
                            , 1
                            , seed
                            , 1
                            , &beta
                            , matrixGrad
                            , opCnfg.unrKrnlCols));
 
                dim3 gd(ceil(opCnfg.unrKrnlCols / 1024.0), 1, 1);
                dim3 bd(1024, 1, 1);
                doKernelRoll<<<gd, bd>>>( matrixGrad
                    , rolledKernelMatrixGrad
                    , opCnfg.kernelRows
                    , opCnfg.kernelCols
                    , opCnfg.unrKrnlCols
                    , opCnfg.unrKrnlRows
                    , opCnfg.colSkip
                    , opCnfg.rowSkip
                    , opCnfg.multiplicandCols + 2 * opCnfg.colPadding
                    , op.cols);
                cudaErrCk( cudaPeekAtLastError() );
                computeGrad(opCnfg.kernel, rolledKernelMatrixGrad);

                cublasErrCk(
                  cublasSgemm( *cublasH
                      , CUBLAS_OP_N
                      , CUBLAS_OP_T
                      , 1
                      , opCnfg.unrKrnlCols
                      , opCnfg.unrKrnlRows
                      , &alpha
                      , seed
                      , 1
                      , unrolledKernel
                      , opCnfg.unrKrnlCols
                      , &beta
                      , colGrad
                      , 1)
                );
                dim3 gd2( ceil(opCnfg.multiplicandCols / 32.0)
                        , ceil(opCnfg.multiplicandRows / 32.0)
                        , 1);
                dim3 bd2(32, 32, 1);
                doCopyWithoutPadding<<<gd2, bd2>>>( colGrad
                        , inputGrad
                        , opCnfg.multiplicandRows
                        , opCnfg.multiplicandCols
                        , opCnfg.rowPadding
                        , opCnfg.colPadding);
                cudaErrCk( cudaPeekAtLastError() );
                computeGrad(opCnfg.multiplicand, inputGrad);
            }break;
            case OperationType::MaxPool: {
                MaxPoolConfig opConfig = get<MaxPoolConfig>(op.config);
                float* targetValue = memLocs[opConfig.target+"_result"];
                float* result = memLocs[op.name+"_result"];
                float* grad = memLocs[op.name+"_grad"];

                dim3 gd(ceil(op.cols / 32.0), ceil(op.rows / 32.0), 1);
                dim3 bd(32, 32, 1);
                doMaxPoolGrad<<<gd, bd>>>(op.rows, op.cols, opConfig.targetRows, opConfig.targetCols, 
                    opConfig.rowSkip, opConfig.height, opConfig.colSkip, opConfig.width, 
                    targetValue, result, seed, grad);
                cudaErrCk( cudaPeekAtLastError() );

                computeGrad(opConfig.target, grad);
            }break;

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

                    cublasErrCk( cublasSgemm( *cublasH
                               , CUBLAS_OP_N
                               , CUBLAS_OP_N
                               , opConfig.target2Cols
                               , opConfig.target1Rows
                               , opConfig.target1Cols
                               , &alpha
                               , d_matrix2
                               , opConfig.target2Cols
                               , d_matrix1
                               , opConfig.target1Cols
                               , &beta
                               , d_result
                               , opConfig.target2Cols) ) 

                break;}

                case OperationType::LeakyReLU:{ 
                    // TODO accessing memLocs like this is quite error prone.
                    // should probably get some accessor functions so I don't make mistakes
                    // the compile can't catch...
                            
                    BasicConfig opConfig = get<BasicConfig>(op.config);
                    float* d_col = memLocs[opConfig.target+"_result"];
                    float* d_grad = memLocs[op.name+"_grad"];
                    float* d_result = memLocs[op.name+"_result"];

                    dim3 bd(32, 32, 1);
                    dim3 gd(ceil(op.cols / 32.0), ceil(op.rows / 32.0), 1);


                    doLeakyReLU<<<gd, bd>>>(op.rows, op.cols, d_grad, d_col, d_result);
                    cudaErrCk( cudaPeekAtLastError() );


                break;}
                case OperationType::Add:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = memLocs[opConfig.target1+"_result"];
                    float* d_v2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    float alpha = 1.0;
                    
                    cublasErrCk( cublasScopy(*cublasH, op.rows * op.cols, d_v1, 1, d_result, 1) );
                    cublasErrCk( cublasSaxpy(*cublasH, op.rows * op.cols, &alpha, d_v2, 1, d_result, 1) );
                break;}
                case OperationType::Scalar:{
                    ScalarConfig opConfig = get<ScalarConfig>(op.config);
                    float* d_target = memLocs[opConfig.target+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    cublasErrCk( cublasScopy(*cublasH, op.rows * op.cols, d_target, 1, d_result, 1) );
                    cublasErrCk( cublasSscal(*cublasH, op.rows * op.cols, &(opConfig.scale), d_result, 1) );
                break;}
                case OperationType::InnerProduct:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = memLocs[opConfig.target1+"_result"];
                    float* d_v2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    cublasErrCk( cublasSdot(*cublasH, opConfig.targetRows, d_v1, 1, d_v2, 1, d_result) );

                break;}
                case OperationType::Convolution:{
                    // pad the input, which means a copy to wocrking, which is slow
                    ConvolutionConfig opCnfg = get<ConvolutionConfig>(op.config);
                    float* input = memLocs[opCnfg.multiplicand+"_result"];
                    float* paddedInput = memLocs[op.name+"_working"];
                    convolutionPadInput( opCnfg.multiplicandRows
                                       , opCnfg.multiplicandCols
                                       , opCnfg.rowPadding
                                       , opCnfg.colPadding
                                       , input
                                       , paddedInput);

                    // unroll the kernel..another slow copy to working.
                    float* kernel = memLocs[opCnfg.kernel+"_result"];
                    float* unrolledKernel = memLocs[op.name+"_working"] + opCnfg.paddedInputSize;
                    convolutionUnrollKernel( opCnfg.unrKrnlRows
                            , opCnfg.unrKrnlCols
                            , opCnfg.kernelRows
                            , opCnfg.kernelCols
                            , opCnfg.multiplicandCols + 2 * opCnfg.colPadding
                            , op.cols
                            , opCnfg.rowSkip
                            , opCnfg.colSkip
                            , kernel
                            , unrolledKernel);

                    // do a matrix multiplication, storing it in the result
                    float alpha = 1;
                    float beta = 0;
                    float* output = memLocs[op.name+"_result"];

                    cublasErrCk( cublasSgemv( *cublasH
                                , CUBLAS_OP_T
                                , opCnfg.unrKrnlCols
                                , opCnfg.unrKrnlRows
                                , &alpha
                                , unrolledKernel
                                , opCnfg.unrKrnlCols
                                , paddedInput
                                , 1
                                , &beta
                                , output
                                , 1 ) );

                break;}
                case OperationType::MaxPool: {
                    MaxPoolConfig opConfig = get<MaxPoolConfig>(op.config);
                    float* d_target = memLocs[opConfig.target+"_result"];
                    float* d_result = memLocs[op.name+"_result"];

                    dim3 gd(ceil(op.cols / 32.0), ceil(op.rows / 32.0), 1);
                    dim3 bd(32, 32, 1);
                    doMaxPool<<<gd, bd>>>(op.rows, op.cols, opConfig.targetRows, opConfig.targetCols,
                        opConfig.rowSkip, opConfig.height, opConfig.colSkip, opConfig.width,
                        d_target, d_result);
                    cudaErrCk( cudaPeekAtLastError() );

                break;}
            }
        }

    }


} // namespace DA
