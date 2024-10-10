#include <iostream>
#include <valarray>
#include <stdexcept>
#include "fast_autodiff.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

namespace FA {


ostream& outputMatrix(ostream& os, float* m, unsigned int rows, unsigned int cols) {
    for(int row {0}; row< rows;row++) {
        for(int col {0}; col<cols;col++) {
            os << m[row*cols + col] << " ";
        }
        os << "\n";
    }
    
    return os;
}

void AD::compute(cublasHandle_t *handle) {}

AD::AD(string _name, unsigned int _rows, unsigned int _cols)
    : name(_name)
    , rows(_rows)
    , cols(_cols) {
    cudaMalloc((void**) &this->d_value, _rows * _cols * sizeof(float));
    cudaMalloc((void**) &this->d_grad, _rows * _cols * sizeof(float));
    this->value = new float[_rows * _cols];
    this->resetGrad();

}

void AD::fromDevice() {
    cudaError_t err;
    int size = this->rows * this->cols * sizeof(float);
    err = cudaMemcpy(this->value, this->d_value, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        printf("cudaMemcpy failed at AD(%s)::fromDevice: %s\n", this->name, cudaGetErrorName(err));
        exit(1);
    }
}

__global__ void doFill( int rows, int cols, float value, float* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * cols + col;
    if( i < rows*cols)
        result[i] = value;
}
void AD::computeGrad(cublasHandle_t *handle) {
    float* seed;
    cudaError_t err;
    int size = this->rows * this->cols * sizeof(float);
    err = cudaMalloc((void**) &seed, size);
    if(err != cudaSuccess) {
        printf("malloc error in Scalar::computeGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    dim3 gd(ceil(this->cols/32.0), ceil(this->rows/32.0), 1);
    dim3 bd(32, 32, 1);
    doFill<<<gd, bd>>>( this->rows, this->cols, 1.0f, seed);


    
    this->pushGrad(handle, seed);
    
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        printf("sync error in Scalar::computeGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }

}

AD::~AD() {
    cudaFree(this->d_grad);
    cudaFree(this->d_value);
    delete [] this->value;
}


    
void AD::resetGrad() {
    cudaMemset(this->d_grad, 0.0, this->rows * this->cols * sizeof(float));
}

void AD::pushGrad(cublasHandle_t *handle, float* d_seed) {
    float alpha = 1;
    cublasSaxpy(*handle, this->cols * this->rows, &alpha, d_seed, 1, this->d_grad, 1);
    cudaFree(d_seed);
}

AbstractCol::AbstractCol(string _name, unsigned int _rows)
    : AD(_name, _rows, 1) {
}


void Col::loadValues(valarray<float> newValues) {
    if(newValues.size() != this->rows)
        throw out_of_range("size of col " + this->name + " (" + to_string(this->rows) + ") does not mathc size of valarray (" + to_string(newValues.size()) + ").");

    cudaMemcpy(this->d_value, &(newValues[0]), this->rows * sizeof(float), cudaMemcpyHostToDevice);

}

Col::Col(string _name, unsigned int _rows)
    : AbstractCol(_name, _rows){
}

__global__
void doGradDescent( float learningRate, int matrixCols, int matrixRows, float* matrix, float* grad) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < matrixRows) && (col < matrixCols)){
        int index = col + matrixCols * row;
        matrix[index] = matrix[index] - (learningRate * grad[index]);
    }


}
void Matrix::gradDescent(cublasHandle_t *handle, float learningRate) {
    float alpha = 1;
    float beta = -1 * learningRate;

    cublasSgeam(*handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                this->rows,
                this->cols,
                &alpha,
                this->d_value,
                this->rows,
                &beta,
                this->d_grad,
                this->rows,
                this->d_value,
                this->rows);
}

void Matrix::loadValues(valarray<float> newValues) {
    if(newValues.size() != this->rows * this->cols)
        throw out_of_range("size of matrix " + this->name + " (" + to_string(this->rows * this->cols) + ") does not match size of valarray (" + to_string(newValues.size()) + ").");
    
    int size = this->rows * this->cols * sizeof(float);

    cudaError_t err = cudaMemcpy(this->d_value, &(newValues[0]), size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        printf("Matrix::loadValues unable to cudaMemcpy: %s - %s\n", 
                cudaGetErrorName(err),
                cudaGetErrorString(err));
        exit(1);
    }


}


Matrix::Matrix(string _name, unsigned int _rows, unsigned int _cols)
    : AD(_name, _rows, _cols) {

}


void MatrixColProduct::resetGrad() {
    AD::resetGrad();
    this->matrix->resetGrad();
    this->col->resetGrad();

}



void MatrixColProduct::pushGrad(cublasHandle_t *handle, float* d_seed) {
    // assert len(seed) == this->matrix->rows

    int matrixSize = this->matrix->rows * this->matrix->cols * sizeof(float);
    int colSize = this->col->rows * sizeof(float);
    float* matrixGrad;
    float* colGrad;

    cudaMalloc((void**) &matrixGrad, matrixSize);
    cudaMalloc((void**) &colGrad, colSize);

    float alpha = 1;
    float beta = 0;

    cublasSgemm(*handle, 
                CUBLAS_OP_T, 
                CUBLAS_OP_N,
                this->matrix->cols,
                this->matrix->rows,
                1,  
                &alpha,
                this->col->d_value,
                1,  
                d_seed,
                1,  
                &beta,
                matrixGrad,
                this->matrix->cols);

    cublasSgemm(*handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                1,
                this->matrix->cols,
                this->matrix->rows,
                &alpha,
                d_seed,
                1,
                this->matrix->d_value,
                this->matrix->cols,
                &beta,
                colGrad,
                1);


    this->matrix->pushGrad(handle, matrixGrad);
    this->col->pushGrad(handle, colGrad);
    cudaFree(d_seed);


}


void MatrixColProduct::compute(cublasHandle_t *handle) {
    this->matrix->compute(handle);
    this->col->compute(handle);

    float alpha = 1;
    float beta = 0;

    cublasSgemv(*handle,
                CUBLAS_OP_T,
                this->matrix->cols,
                this->matrix->rows,
                &alpha,
                this->matrix->d_value,
                this->matrix->cols,
                this->col->d_value,
                1,
                &beta,
                this->d_value,
                1);



}

MatrixColProduct::MatrixColProduct(AD* m, AbstractCol* x)
    : AbstractCol("Matrix product of " + m->name + " and " + x->name, m->rows)
    , matrix(m)
    , col(x) {

    if(m->cols != x->rows)
        throw invalid_argument("Input matrix " + m->name
                              + " has " + to_string(m->cols) + " rows but"
                              + " column vector has " + to_string(x->rows) 
                              + " columns.");

}

MatrixColProduct::~MatrixColProduct() {
    delete this->matrix;
    delete this->col;
}

void ColLeakyReLU::resetGrad() {
    AD::resetGrad();
    this->col->resetGrad();
}

__global__
void doComponentProduct( int rows
                       , float* grad
                       , float* seed
                       , float* result ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if( row < rows) {
        result[row] = seed[row] * grad[row];
    }

}

void ColLeakyReLU::pushGrad(cublasHandle_t *handle, float* d_seed) {
    float* newSeed;
    cudaError_t err;

    cudaMalloc((void**) &newSeed, this->rows * sizeof(float));

    dim3 bd(1, 1024, 1);
    dim3 gd(1, ceil((this->col->rows)/1024.0), 1);

    doComponentProduct<<<gd, bd>>>(this->rows, this->d_grad, d_seed, newSeed);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in ColLeakyReLU::pushGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaDeviceSynchronize();

    this->col->pushGrad(handle, newSeed);
    cudaFree(d_seed);
}


__global__
void doLeakyReLU( int Arows
                , int Acols
                , float* grad
                , float* A 
                , float* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < Arows) && (col < Acols)) {
        int i = row * Acols + col;
        if (A[i] > 0) {
            grad[i] = 1;
            result[i] = A[i];
        }else {
            result[i] = 0.01 * A[i];
            grad[i] = 0.01;
        }
    }
}

void ColLeakyReLU::compute(cublasHandle_t *handle) {
    this->col->compute(handle);
    cudaError_t err;

    dim3 bd (1, 1024, 1);
    dim3 gd (1, ceil((this->col->rows)/1024.0), 1);

    doLeakyReLU<<<gd, bd>>>( this->col->rows, 1, this->d_grad, this->col->d_value, this->d_value);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in ColLeakyReLU::compute: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();


}

ColLeakyReLU::ColLeakyReLU(AbstractCol* _col)
    : col(_col)
    , AbstractCol( "ReLU of " + _col->name, _col->rows) {
}

ColLeakyReLU::~ColLeakyReLU() {
    delete this->col;
}

void Scalar::resetGrad() {
    AD::resetGrad();
    this->col->resetGrad();
}



void Scalar::pushGrad(cublasHandle_t *handle, float* d_seed) {
    float *newSeed;
    cudaMalloc((void**) &newSeed, this->col->rows * sizeof(float));
    cublasScopy(*handle, this->col->rows, d_seed, 1, newSeed, 1);
    cublasSscal(*handle, this->col->rows, &(this->scalar), newSeed, 1);

    cudaFree(d_seed);
    this->col->pushGrad(handle, newSeed);
}


void Scalar::compute(cublasHandle_t *handle) {
    this->col->compute(handle);

    cublasScopy(*handle, this->col->rows, this->col->d_value, 1, this->d_value, 1);
    cublasSscal(*handle, this->col->rows, &(this->scalar), this->d_value,1);

}

Scalar::Scalar(AbstractCol* _col, float _scalar)
    : col(_col)
    , scalar(_scalar)
    , AbstractCol( "Scalar of (" + _col->name + ") by "+ to_string(_scalar), _col->rows) {
}

Scalar::~Scalar() {
    delete this->col;
}


void AddCol::resetGrad() {
    AD::resetGrad();
    this->col1->resetGrad();
    this->col2->resetGrad();
}

void AddCol::pushGrad(cublasHandle_t *handle, float* d_seed) {
    float* copySeed ;
    cudaMalloc((void**) &copySeed, this->col1->rows * sizeof(float));
    cudaMemcpy(copySeed, d_seed, this->col1->rows * sizeof(float), cudaMemcpyDeviceToDevice);
    this->col1->pushGrad(handle, d_seed);
    this->col2->pushGrad(handle, copySeed);
}


void AddCol::compute(cublasHandle_t *handle) {
    this->col1->compute(handle);
    this->col2->compute(handle);
    float alpha = 1;

    cublasScopy(*handle, this->col1->rows, this->col1->d_value, 1, this->d_value, 1);

    cublasSaxpy(*handle, this->col1->rows, &alpha, this->col2->d_value, 1, this->d_value, 1);



}

AddCol::AddCol(AbstractCol* _col1, AbstractCol* _col2) 
    : col1(_col1)
    , col2(_col2)
    , AbstractCol("Sum of (" + _col1->name + ") and (" + _col2->name + ")", _col1->rows) {
}

AddCol::~AddCol() {
    if( this->col1 == this->col2) {
        delete this->col1;
    } else {
        delete this->col1;
        delete this->col2;
    }
}

void InnerProduct::resetGrad() {
    AD::resetGrad();
    this->col1->resetGrad();
    this->col2->resetGrad();
}

void InnerProduct::pushGrad(cublasHandle_t *handle, float* d_seed) {
    // assume len(seed)=1 here...
    float* vec1;
    float* vec2;
    float* scalar = new float;

    cudaMemcpy(scalar, d_seed, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMalloc((void**) &vec1, this->col1->rows * sizeof(float));

    cudaMalloc((void**) &vec2, this->col2->rows * sizeof(float));


    cublasScopy(*handle, this->col1->rows, this->col1->d_value, 1, vec1, 1);
    cublasSscal(*handle, this->col1->rows, scalar, vec1,1);

    cublasScopy(*handle, this->col2->rows, this->col2->d_value, 1, vec2, 1);
    cublasSscal(*handle, this->col2->rows, scalar, vec2,1);

    
    this->col2->pushGrad(handle, vec1);
    this->col1->pushGrad(handle, vec2);

    delete scalar;

    cudaFree(d_seed);
}

void InnerProduct::compute(cublasHandle_t *handle) {
    this->col1->compute(handle);
    this->col2->compute(handle);

    cublasSdot(*handle, this->col1->rows, this->col1->d_value, 1,
               this->col2->d_value, 1, this->d_value);

}

InnerProduct::InnerProduct(AbstractCol* _col1, AbstractCol* _col2) 
    : col1(_col1)
    , col2(_col2)
    , AbstractCol("Inner Product of (" + _col1->name + ") and (" + _col2->name + ")", 1) {
}

InnerProduct::~InnerProduct() {
    if( this->col1 == this->col2) {
        delete this->col1;
    } else {
        delete this->col1;
        delete this->col2;
    }
}

__global__ void doPadInput(float* input, float* paddedInput,
        int inputRows, int inputCols, int rowPadding, int colPadding) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = inputRows + 2*rowPadding;
    int cols = inputCols + 2*colPadding;

    if(row < rows && col < cols) {
        if(row - rowPadding >= 0 && row < inputRows +rowPadding
            && col - colPadding >= 0 && col < inputCols + colPadding)
            paddedInput[row*cols + col] = input[(row - rowPadding) * inputCols + col - colPadding];
        else
            paddedInput[row*cols + col] = 0;
    }
}

void Convolution::padInput() {
    int paddedInputRows = this->multiplicand->rows + this->rowPadding * 2;
    int paddedInputCols = this->multiplicand->cols + this->colPadding * 2;

    cudaError_t err; 
    dim3 gd(ceil(paddedInputCols/32.0), ceil(paddedInputRows/32.0), 1);
    dim3 bd(32, 32, 1);
    doPadInput<<<gd, bd>>>(this->multiplicand->d_value, this->d_input,
            this->multiplicand->rows, this->multiplicand->cols, 
            this->rowPadding, this->colPadding);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in Convolution::padInput: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();




}


__global__ void doUnroll(float* kernel, float* matrix,
                         int kernelRows, int kernelCols,
                         int mRows, int mCols,
                         int inCols, int outCols,
                         int rowSkip, int colSkip) {
    int mrow = blockIdx.y * blockDim.y + threadIdx.y;
    int mcol = blockIdx.x * blockDim.x + threadIdx.x;
    if(mrow < mRows && mcol < mCols) {
        int outRow = mrow / outCols;
        int outCol = mrow % outCols;

        int inRow = mcol / inCols;
        int inCol = mcol % inCols;

        int kRowIndex = inRow - rowSkip * outRow;
        int kColIndex = inCol - colSkip * outCol;

        if(    kRowIndex >= 0 && kRowIndex < kernelRows 
            && kColIndex >=0 && kColIndex < kernelCols) {
            matrix[mrow*mCols + mcol] = kernel[kRowIndex * kernelCols + kColIndex];
        }else{
            matrix[mrow*mCols + mcol] = 0;
        }

    }
}

void Convolution::unrollKernel() {


    cudaError_t err; 
    dim3 gd(ceil(this->unrKrnlCols/32.0), ceil(this->unrKrnlRows/32.0), 1);
    dim3 bd(32, 32, 1);
    doUnroll<<<gd, bd>>>(this->kernel->d_value, this->d_kernel,
             this->kernel->rows, this->kernel->cols,
             this->unrKrnlRows, this->unrKrnlCols,
             this->multiplicand->cols + 2*this->colPadding, this->cols,
             this->rowSkip, this->colSkip);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in Convolution::unrollKerenl: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();
}

void Convolution::resetGrad() {
    AD::resetGrad();
    this->multiplicand->resetGrad();
    this->kernel->resetGrad();
}


// __global__ void doConvolution(int targetRows, int targetCols,
//                               int multiplicandRows, int multiplicandCols,
//                               int rowPadding, int rowSkip, int kernelRows,
//                               int colPadding, int colSkip, int kernelCols,
//                               float* multiplicand,
//                               float* kernel,
//                               float* result) {
//     int trow = blockIdx.y * blockDim.y + threadIdx.y;
//     int tcol = blockIdx.x * blockDim.x + threadIdx.x;
//     if(trow < targetRows && tcol < targetCols) {
//         float val = 0;
//         int mrow = - rowPadding + rowSkip * trow;
//         int mcol = - colPadding + colSkip * tcol;
// 
//         for(int i =mrow; i < mrow + kernelRows; i++)
//             for(int j =mcol; j< mcol + kernelCols; j++) 
//                 if(i >= 0 && j >= 0 && i < multiplicandRows && j < multiplicandCols) {
//                     int mIndex = multiplicandCols * i + j;
//                     int kIndex = kernelCols * (i-mrow) + (j-mcol);   
//                     val += multiplicand[mIndex] * kernel[kIndex];
//                 }
//         
// 
//         result[targetCols * trow + tcol] = val;
//     }
// }
// 
// 
// void Convolution::compute(cublasHandle_t *handle) {
//     this->multiplicand->compute(handle);
//     this->kernel->compute(handle);
//     cudaError_t err; 
//     dim3 gd(ceil(this->cols/32.0), ceil(this->rows/32.0), 1);
//     dim3 bd(32, 32, 1);
//     doConvolution<<<gd, bd>>>(this->rows, this->cols,
//                               this->multiplicand->rows, this->multiplicand->cols,
//                               this->rowPadding, this->rowSkip, this->kernel->rows,
//                               this->colPadding, this->colSkip, this->kernel->cols,
//                               this->multiplicand->d_value,
//                               this->kernel->d_value,
//                               this->d_value );
//     err = cudaGetLastError();
//     if(err != cudaSuccess) {
//         printf("Kernel launch error in Convolution::compute: %s\n", cudaGetErrorString(err));
//         exit(1);
//     }
// 
//     cudaDeviceSynchronize();
// }
//
__global__ void doKernelRoll(float* matrix, float* kernel,
                         int kernelRows, int kernelCols,
                         int mCols, int mRows,
                         int colSkip, int rowSkip,
                         int inCols, int outCols) {
    int mcol = blockIdx.x * blockDim.x + threadIdx.x;
    if(mcol < mCols) {
        int kRow = mcol / inCols;
        int kCol = mcol % inCols;
        if(    kRow >= 0 && kRow < kernelRows 
            && kCol >=0 && kCol < kernelCols) {
            float val = 0;
            for(int mrow = 0; mrow<mRows; mrow++) {
                int ocol = mrow % outCols;
                int orow = mrow / outCols;

                int offset = colSkip * ocol + rowSkip * orow * inCols;

                val += matrix[mrow*mCols + mcol + offset];

            }
            kernel[kRow * kernelCols + kCol] = val;
        }
    }
}

__global__ void doCopyWithoutPadding(float* paddedSource, float* dest,
        int rows, int cols, int rowPadding, int colPadding) {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     if(row < rows && col <cols) {
         dest[row*cols + col] = paddedSource[(row+ rowPadding)*(cols + 2* colPadding) + col + colPadding];
     }
}
                                    

void Convolution::pushGrad(cublasHandle_t *handle, float* d_seed) {
    // assert len(seed) == this->matrix->rows

    int matrixSize = this->unrKrnlRows * this->unrKrnlCols * sizeof(float);
    int kernelSize = this->kernel->cols * this->kernel->rows * sizeof(float);
    int colSize = this->unrKrnlCols * sizeof(float);
    int inputSize = this->multiplicand->cols * this->multiplicand->rows * sizeof(float);
    float *rolledKernelGrad, *matrixGrad, *colGrad, *inputGrad;


    float alpha = 1;
    float beta = 0;

    cudaMalloc((void**) &rolledKernelGrad, kernelSize);
    cudaMalloc((void**) &matrixGrad, matrixSize);
    cublasSgemm(*handle, 
                CUBLAS_OP_T, 
                CUBLAS_OP_N,
                this->unrKrnlCols,
                this->unrKrnlRows,
                1,  
                &alpha,
                this->d_input,
                1,  
                d_seed,
                1,  
                &beta,
                matrixGrad,
                this->unrKrnlCols);
    dim3 gd(ceil(this->unrKrnlCols/1024.0), 1, 1);
    dim3 bd(1024, 1, 1);
    doKernelRoll<<<gd, bd>>>(matrixGrad, rolledKernelGrad,
                  this->kernel->rows, this->kernel->cols,
                  unrKrnlCols, unrKrnlRows,
                  colSkip, rowSkip,
                  this->multiplicand->cols + 2*this->colPadding,
                  this->cols);
    cudaFree(matrixGrad);
    this->kernel->pushGrad(handle, rolledKernelGrad);

    cudaMalloc((void**) &colGrad, colSize);
    cudaMalloc((void**) &inputGrad, inputSize);
    cublasSgemm(*handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                1,
                this->unrKrnlCols,
                this->unrKrnlRows,
                &alpha,
                d_seed,
                1,
                this->d_kernel,
                this->unrKrnlCols,
                &beta,
                colGrad,
                1);
    dim3 gd2(ceil(this->multiplicand->cols/32.0), ceil(this->multiplicand->rows/32.0),1);
    dim3 bd2(32, 32, 1);
    doCopyWithoutPadding<<<gd2,bd2>>>(colGrad, inputGrad, this->multiplicand->rows, this->multiplicand->cols, this->rowPadding, this->colPadding);
    cudaFree(colGrad);
    this->multiplicand->pushGrad(handle, inputGrad);



    cudaFree(d_seed);


}

void Convolution::compute(cublasHandle_t *handle) {
    this->multiplicand->compute(handle);
    this->kernel->compute(handle);
    this->padInput();
    this->unrollKernel();
    float alpha = 1;
    float beta = 0;

    cublasSgemv(*handle,
                CUBLAS_OP_T,
                this->unrKrnlCols,
                this->unrKrnlRows,
                &alpha,
                this->d_kernel,
                this->unrKrnlCols,
                this->d_input,
                1,
                &beta,
                this->d_value,
                1);

}


Convolution::Convolution(AD* m, AD* k,
        unsigned int rowPadding, unsigned int rowSkip,
        unsigned int colPadding, unsigned int colSkip)
    : multiplicand(m)
    , kernel(k)
    , rowPadding(rowPadding)
    , rowSkip(rowSkip)
    , colPadding(colPadding)
    , colSkip(colSkip)
    , AD("Convolution of "+m->name, (m->rows + 2 * rowPadding - k->rows)/rowSkip + 1, (m->cols + 2*colPadding - k->cols)/colSkip + 1){  

    this->unrKrnlCols = (this->multiplicand->rows + this->rowPadding *2) 
                             * (this->multiplicand->cols + this->colPadding * 2);
    this->unrKrnlRows = this->rows * this->cols;
    unsigned int unrKrnlSize = this->unrKrnlRows * this->unrKrnlCols;
    cudaMalloc((void**) &this->d_kernel, unrKrnlSize * sizeof(float));
    cudaMalloc((void**) &this->d_input, this->unrKrnlCols * sizeof(float));
}

Convolution::~Convolution() {
    delete this->multiplicand;
    delete this->kernel;
    cudaFree(this->d_kernel);
    cudaFree(this->d_input);
}

}
