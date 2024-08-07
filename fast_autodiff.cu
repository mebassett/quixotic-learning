#include <iostream>
#include<valarray>
#include "fast_autodiff.h"

using namespace std;

namespace FA {

__global__ void doAdd( int Arows, int Acols, float* result, float* A, float* B ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < Arows) && (col < Acols)) {
        int i = row * Acols + col;
        result[i] = A[i] + B[i];
    }
}

__global__ void doScale( int Arows, int Acols, float* result, float* A, float scalar) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < Arows) && (col < Acols)) {
        int i = row * Acols + col;
        result[i] = A[i] * scalar;
    }
}



void AD::compute() {}

AD::AD(string _name, unsigned int _rows, unsigned int _cols)
    : name(_name)
    , rows(_rows)
    , cols(_cols) {
    cudaMalloc((void**) &this->d_grad, _rows * _cols * sizeof(float));

}


    
void AD::resetGrad() {
    cudaMemset(this->d_grad, 0.0, this->rows*sizeof(float));
}

void AD::pushGrad(float* d_seed) {
    cudaError_t err;

    dim3 gd(ceil(this->cols/32.0), ceil(this->rows/32.0), 1);
    dim3 bd(32, 32, 1);

    doAdd<<<gd, bd>>>( this->rows, this->cols, this->d_grad, this->d_grad, d_seed );
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in AD pushGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }

}

AbstractCol::AbstractCol(string _name, unsigned int _rows)
    : AD(_name, _rows, 1) {
    int size = _rows * sizeof(float);
    cudaMalloc((void**) &this->d_value, size);
    //cudaMallocManaged(&this->value, _rows * sizeof(float));
    //cudaMallocManaged(&this->grad, _rows * sizeof(float));
}

AbstractCol::~AbstractCol() {
    //cudaFree(this->value);
    //cudaFree(this->grad);
    cudaFree(this->d_value);
    cudaFree(this->d_grad);
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
void Matrix::gradDescent(float learningRate) {
    cudaError_t err;

    dim3 gd(ceil(this->cols/32.0), ceil(this->rows/32.0), 1);
    dim3 bd(32, 32, 1);
    doGradDescent<<< gd, bd >>> (learningRate, this->cols, this->rows, this->d_value, this->d_grad);

    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in Matrix::gradDescent: %s - %s\n", 
                cudaGetErrorName(err),
                cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();
}

void Matrix::loadValues(valarray<float> newValues) {
    if(newValues.size() != this->rows * this->cols)
        throw out_of_range("size of matrix " + this->name + " (" + to_string(this->rows * this->cols) + ") does not match size of valarray (" + to_string(newValues.size()) + ").");
    
    int size = this->rows * this->cols * sizeof(float);

    cudaMemcpy(this->d_value, &(newValues[0]), size, cudaMemcpyHostToDevice);

}


Matrix::Matrix(string _name, unsigned int _rows, unsigned int _cols)
    : AD(_name, _rows, _cols) {
    int size = _cols * _rows * sizeof(float);
    cudaMalloc((void**) &this->d_value, size);

}

Matrix::~Matrix() {
    //cudaFree(this->value);
    //cudaFree(this->grad);
    delete [] this->value;
    cudaFree(this->d_value);
    cudaFree(this->d_grad);
}

void MatrixColProduct::resetGrad() {
    AbstractCol::resetGrad();
    this->matrix->resetGrad();
    this->col->resetGrad();

}

void MatrixColProduct::fromDevice() {
    cudaError_t err;
    int size = this->rows * sizeof(float);
    err = cudaMemcpy(this->value, this->d_value, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        printf("cudaMemcpy failed at MatrixColProduct::fromDevice: %s\n", cudaGetErrorName(err));
        exit(1);
    }
}

//__global__
//void doMatrixColProductGrad( float* colGrad
//                           , float* matGrad
//                           , float* seed
//                           , float* A
//                           , float* B ) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    int index = col + Acols * row;
//    matGrad[index] = seed[row] * B[col];
//    colGrad[col] += seed[row] * A[index]; 
//}
__global__
void doMatrixColGrad( int matrixCols
                    , int matrixRows
                    , float* matrix
                    , float* column
                    , float* d_seed
                    , float* matrixGrad) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < matrixRows) && (col < matrixCols)) {
        int index = col + matrixCols * row;
        matrixGrad[index] = d_seed[row] * column[col];
    }

}

void MatrixColProduct::pushGrad(float* d_seed) {
    // assert len(seed) == this->matrix->rows

    int size = this->matrix->rows * this->matrix->cols * sizeof(float);
    float* matrixGrad;
    cudaError_t err;

    cudaMalloc((void**) &matrixGrad, size);

    dim3 gd(ceil(this->matrix->cols/32.0), ceil(this->matrix->rows/32.0), 1);
    dim3 bd(32, 32, 1);

    doMatrixColGrad<<< gd, bd>>>( this->matrix->cols, this->matrix->rows 
                                , this->matrix->d_value
                                , this->col->d_value, d_seed, matrixGrad );

    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error MatrixColProduct::pushGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaDeviceSynchronize();

    this->matrix->pushGrad(matrixGrad);
    //this->col->pushGrad(colGrad);


}

__global__
void doMatrixProduct( int Arows // Acols = Brows
                    , int Acols
                    , int Bcols
                    , float* result
                    , float* A
                    , float* B ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < Arows) && (col < Bcols)) {
        float val = 0;
        for(int i {0}; i < Acols; i ++) 
            val += A[(row * Acols) + i] * B[ (i * Bcols) + col  ];

        result[(row*Bcols) + col] = val;
    }
}

void MatrixColProduct::compute() {
    this->matrix->compute();
    this->col->compute();
    cudaError_t err;

    dim3 bd(1, 1024, 1);
    dim3 gd(1, ceil((this->col->rows)/1024.0), 1);


    doMatrixProduct<<<gd, bd>>>( this->matrix->rows, this->matrix->cols, 1
                               , this->d_value, this->matrix->d_value
                               , this->col->d_value);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in MatrixColProduct::compute: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        printf("sync error in MatrixColProduct::compute: %s\n", cudaGetErrorString(err));
        exit(1);
    }


}

MatrixColProduct::MatrixColProduct(Matrix* m, AbstractCol* x)
    : AbstractCol("Matrix product of " + m->name + " and " + x->name, m->rows)
    , matrix(m)
    , col(x) {
    this->value = new float[x->rows];
}

MatrixColProduct::~MatrixColProduct() {
    delete [] this->value;
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

void ColLeakyReLU::pushGrad(float* d_seed) {
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

    this->col->pushGrad(newSeed);
}


__global__
void doLeakyReLU( int Arows
                , int Acols
                , float* grad
                , float* A ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < Arows) && (col < Acols)) {
        int i = row * Acols + col;
        if (A[i] > 0) {
            grad[i] = 1;
        }else {
            A[i] = 0.01 * A[i];
            grad[i] = 0.01;
        }
    }
}

void ColLeakyReLU::compute() {
    this->col->compute();
    cudaError_t err;

    dim3 bd (1, 1024, 1);
    dim3 gd (1, ceil((this->col->rows)/1024.0), 1);

    doLeakyReLU<<<gd, bd>>>( this->col->rows, 1, this->d_grad, this->d_value);
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

void Scalar::resetGrad() {
    AD::resetGrad();
    this->col->resetGrad();
}

void Scalar::fromDevice() {
    cudaError_t err;
    err = cudaMemcpy(this->value, this->d_value, this->col->rows * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        printf("cudaMemcpy failed at Scalar::fromDevice: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void Scalar::computeGrad() {
    float* seed;
    cudaError_t err;
    err = cudaMalloc((void**) &seed, this->col->rows * sizeof(float));
    if(err != cudaSuccess) {
        printf("malloc error in Scalar::computeGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemset(seed, 1.0, this->col->rows * sizeof(float));
    if(err != cudaSuccess) {
        printf("memset error in Scalar::computeGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    this->pushGrad(seed);
    
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        printf("sync error in Scalar::computeGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }

}

void Scalar::pushGrad(float* d_seed) {
    float *newSeed;
    cudaError_t err;

    cudaMalloc((void**) &newSeed, this->col->rows * sizeof(float));

    dim3 bd (1, 1024, 1);
    dim3 gd (1, ceil((this->col->rows)/1024.0), 1);

    doScale<<<gd, bd>>>( this->col->rows, 1, newSeed, this->d_value, this->scalar );
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in Scalar::pushGrad: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    this->col->pushGrad(newSeed);
}


void Scalar::compute() {
    this->col->compute();
    cudaError_t err;


    dim3 bd (1, 1024, 1);
    dim3 gd (1, ceil((this->col->rows)/1024.0), 1);

    doScale<<<gd, bd>>>( this->col->rows, 1, this->d_value, this->col->d_value, this->scalar );
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in Scalar::compute: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaDeviceSynchronize();


}

Scalar::Scalar(AbstractCol* _col, float _scalar)
    : col(_col)
    , scalar(_scalar)
    , AbstractCol( "Scalar of (" + _col->name + ") by "+ to_string(_scalar), _col->rows) {
    this->value = new float[_col->rows];
}

Scalar::~Scalar() {
    delete [] this->value;
}

void AddCol::resetGrad() {
    AD::resetGrad();
    this->col1->resetGrad();
    this->col2->resetGrad();
}

void AddCol::pushGrad(float* d_seed) {
    this->col1->pushGrad(d_seed);
    this->col2->pushGrad(d_seed);
}


void AddCol::compute() {
    this->col1->compute();
    this->col2->compute();
    cudaError_t err;


    dim3 bd (1, 1024, 1);
    dim3 gd (1, ceil((this->col1->rows)/1024.0), 1);

    doAdd<<<gd, bd>>>( this->col1->rows, 1, this->d_value, this->col1->d_value, this->col2->d_value );
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error in AddCol::compute: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaDeviceSynchronize();



}

AddCol::AddCol(AbstractCol* _col1, AbstractCol* _col2) 
    : col1(_col1)
    , col2(_col2)
    , AbstractCol("Sum of (" + _col1->name + ") and (" + _col2->name + ")", _col1->rows) {
}

void InnerProduct::resetGrad() {
    AD::resetGrad();
    this->col1->resetGrad();
    this->col2->resetGrad();
}

void InnerProduct::pushGrad(float* d_seed) {
    // assume len(seed)=1 here...
    float* vec1;
    float* vec2;
    float* scalar = new float;
    cudaError_t err;

    cudaMemcpy(scalar, d_seed, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMalloc((void**) &vec1, this->col1->rows * sizeof(float));

    cudaMalloc((void**) &vec2, this->col2->rows * sizeof(float));

    dim3 bd (1, 1024, 1);
    dim3 gd (1, ceil((this->col1->rows)/1024.0), 1);
    doScale<<<gd, bd>>>( this->col1->rows, 1, vec1, this->col1->d_value, *scalar );
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error InnerProduct::pushGrad (col1): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    doScale<<<gd, bd>>>( this->col1->rows, 1, vec2, this->col2->d_value, *scalar );
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error InnerProduct::pushGrad (col2): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        printf("sync error InnerProduct::pushGrad (col2): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    this->col2->pushGrad(vec1);
    this->col1->pushGrad(vec2);

    delete scalar;
}

__global__
void doSingleSum(int rows, float* arr, float* result) {
    float sum = 0;
    for(int i {0} ; i < rows; i++) {
        sum += arr[i];
    }
    *result = sum;
}

void InnerProduct::compute() {
    this->col1->compute();
    this->col2->compute();

    float* product;
    cudaError_t err;

    cudaMalloc((void**) &product, this->col1->rows * sizeof(float));

    dim3 bd(1, 1024, 1);
    dim3 gd(1, ceil((this->col1->rows)/1024.0), 1);


    doComponentProduct<<<gd, bd>>>(this->col1->rows, this->col1->d_value
                                  , this->col2->d_value, product);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error InnerProduct::compute (component): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        printf("InnerProduct::compute could not sync device: %s\n", cudaGetErrorName(err));
        exit(1);
    }

    doSingleSum<<<1, 1>>>(this->col1->rows, product, this->d_value);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error InnerProduct::compute(sum): %s - %s \n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
    cudaDeviceSynchronize();

}

InnerProduct::InnerProduct(AbstractCol* _col1, AbstractCol* _col2) 
    : col1(_col1)
    , col2(_col2)
    , AbstractCol("Inner Product of (" + _col1->name + ") and (" + _col2->name + ")", 1) {
}

}
