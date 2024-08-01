#include <iostream>
#include<valarray>
#include "fast_autodiff.h"

using namespace std;

namespace FA {


void AD::resetGrad() {
    for(int i {0}; i< this->rows * this->cols; i++)
        *(this->grad+i) = 0;
}

void AD::pushGrad(float seed[]) {
    for(int i {0}; i< this->rows * this->cols; i++)
        *(this->grad + i) += seed[i];
}

void AD::compute() {}

AD::AD(string _name, unsigned int _rows, unsigned int _cols)
    : name(_name)
    , rows(_rows)
    , cols(_cols)
{}


    

AbstractCol::AbstractCol(string _name, unsigned int _rows)
    : AD(_name, _rows, 1) {
    this->value = new float[_rows];
    this->grad = new float[_rows];
    //cudaMallocManaged(&this->value, _rows * sizeof(float));
    //cudaMallocManaged(&this->grad, _rows * sizeof(float));
    for(int i {0}; i<_rows; i++)
        *(this->grad+i) = 0;
}

AbstractCol::~AbstractCol() {
    //cudaFree(this->value);
    //cudaFree(this->grad);
    delete [] this->value;
    delete [] this->grad;
}

void Col::loadValues(valarray<float> newValues) {
    if(newValues.size() != this->rows)
        throw out_of_range("size of col " + this->name + " (" + to_string(this->rows) + ") does not mathc size of valarray (" + to_string(newValues.size()) + ").");

    for(int i {0}; i< newValues.size();i++)
        *(this->value + i) = newValues[i];
}

Col::Col(string _name, unsigned int _rows)
    : AbstractCol(_name, _rows){
}



void Matrix::loadValues(valarray<float> newValues) {
    if(newValues.size() != this->rows * this->cols)
        throw out_of_range("size of matrix " + this->name + " (" + to_string(this->rows * this->cols) + ") does not match size of valarray (" + to_string(newValues.size()) + ").");

    for(int i {0}; i< newValues.size();i++)
        *(this->value + i) = newValues[i];
}


Matrix::Matrix(string _name, unsigned int _rows, unsigned int _cols)
    : AD(_name, _rows, _cols) {
    this->value = new float[_rows * _cols];
    this->grad = new float[_rows * _cols];
    //cudaMallocManaged(&this->value, _rows * _cols * sizeof(float));
    //cudaMallocManaged(&this->grad, _rows * _cols * sizeof(float));
    for(int i {0}; i<_rows*_cols; i++)
        *(this->grad+i) = 0;

}

Matrix::~Matrix() {
    //cudaFree(this->value);
    //cudaFree(this->grad);
    delete [] this->value;
    delete [] this->grad;
}

void MatrixColProduct::resetGrad() {
    AD::resetGrad();
    this->matrix->resetGrad();
    this->col->resetGrad();

}

void MatrixColProduct::pushGrad(float seed[]) {
    // assert len(seed) == this->matrix->rows

    int size = this->matrix->rows * this->matrix->cols;
    float matGrad[size];
    float colGrad[this->col->rows] = {0};
    for(int i {0}; i< this->matrix->rows; i++){
        for(int j {0}; j< this->matrix->cols; j++) {
            int index = j + this->matrix->cols * i;
            matGrad[index] = seed[i] * *(this->col->value + j);
            colGrad[j] += seed[i] * *(this->matrix->value + index);
        }
    }
    this->matrix->pushGrad(matGrad);
    this->col->pushGrad(colGrad);


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
    delete [] this->value;

    float *A, *B, *result;
    int matrix_size = this->matrix->cols * this->matrix->rows * sizeof(float);
    int col_size = this->col->rows * sizeof(float);

    cudaMalloc((void**) &A, matrix_size);
    cudaMalloc((void**) &B, col_size);
    cudaMalloc((void**) &result, col_size);

    cudaMemcpy(A, this->matrix->value, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, this->col->value, col_size, cudaMemcpyHostToDevice);
    

    dim3 bd(1, 1024, 1);
    dim3 gd(1, ceil((this->col->rows)/1024.0), 1);


    doMatrixProduct<<<gd, bd>>>( this->matrix->rows, this->matrix->cols, 1
                               , result, A, B);
    cudaDeviceSynchronize();

    this->value = new float[this->col->rows];
    cudaMemcpy(this->value, result, col_size, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(result);


}

MatrixColProduct::MatrixColProduct(Matrix* m, AbstractCol* x)
    : AbstractCol("Matrix product of " + m->name + " and " + x->name, m->rows)
    , matrix(m)
    , col(x) {
}

void ColLeakyReLU::resetGrad() {
    AD::resetGrad();
    this->col->resetGrad();
}

void ColLeakyReLU::pushGrad(float seed[]) {
    float newSeed[this->col->rows];
    for(int i {0}; i< this->col->rows; i++)
        newSeed[i] = seed[i] * *(this->grad + i);
    
    this->col->pushGrad(newSeed);
}


void ColLeakyReLU::compute() {
    this->col->compute();

    for(int i {0}; i<this->col->rows; i++) {
        if(*(this->col->value + i) > 0) {
            *(this->value +i) = *(this->col->value +i);
            *(this->grad + i) = 1;
        } else {
            *(this->value +i) = 0.01 * *(this->col->value +i);
            *(this->grad + i) = 0.01;
        }
    }


}

ColLeakyReLU::ColLeakyReLU(AbstractCol* _col)
    : col(_col)
    , AbstractCol( "ReLU of " + _col->name, _col->rows) {
}

void Scalar::resetGrad() {
    AD::resetGrad();
    this->col->resetGrad();
}

void Scalar::pushGrad(float seed[]) {
    float newSeed[this->col->rows];

    for(int i {0}; i< this->col->rows; i++)
        newSeed[i] = seed[i] * this->scalar;
    
    this->col->pushGrad(newSeed);
}

void Scalar::compute() {
    this->col->compute();

    for(int i {0}; i<this->col->rows; i++) {
        *(this->value +i) = this->scalar * *(this->col->value +i);
    }
}

Scalar::Scalar(AbstractCol* _col, float _scalar)
    : col(_col)
    , scalar(_scalar)
    , AbstractCol( "Scalar of (" + _col->name + ") by "+ to_string(_scalar), _col->rows) {
}

void AddCol::resetGrad() {
    AD::resetGrad();
    this->col1->resetGrad();
    this->col2->resetGrad();
}

void AddCol::pushGrad(float seed[]) {
    this->col1->pushGrad(seed);
    this->col2->pushGrad(seed);
}


void AddCol::compute() {
    this->col1->compute();
    this->col2->compute();

    for(int i {0}; i< this->col1->rows;i++) {
        *(this->value + i) = this->col1->value[i] + this->col2->value[i];
    }

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

void InnerProduct::pushGrad(float seed[]) {
    // assume len(seed)=1 here...
    float vec1[this->col1->rows];
    float vec2[this->col2->rows];

    for(int i {0}; i< this->col1->rows; i++)
        vec1[i] = seed[0] * *(this->col1->value + i);
    
    this->col2->pushGrad(vec1);

    for(int i {0}; i< this->col2->rows; i++)
        vec2[i] = seed[0] * *(this->col2->value + i);
    
    this->col1->pushGrad(vec2);
}


void InnerProduct::compute() {
    this->col1->compute();
    this->col2->compute();

    float sum = 0;

    for(int i {0}; i< this->col1->rows;i++) {
        sum += this->col1->value[i] * this->col2->value[i];
    }
    *(this->value) = sum;

}

InnerProduct::InnerProduct(AbstractCol* _col1, AbstractCol* _col2) 
    : col1(_col1)
    , col2(_col2)
    , AbstractCol("Inner Product of (" + _col1->name + ") and (" + _col2->name + ")", 1) {
}

}
