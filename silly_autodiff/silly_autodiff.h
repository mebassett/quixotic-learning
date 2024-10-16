// silly_autodiff.h
#ifndef FAST_AUTODIFF_H
#define FAST_AUTODIFF_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <valarray>
#include <vector>

namespace FA {

std::ostream& outputMatrix(std::ostream& os, float* m, unsigned int rows,
    unsigned int cols);

struct AD {
    std::string name;
    bool initCuda;
    unsigned int rows;
    unsigned int cols;
    float* d_grad;
    float* d_value;
    float* value;
    void getPartials(float* p);
    virtual void fromDevice();
    virtual void compute(cublasHandle_t* handle);
    virtual void resetGrad();
    virtual void pushGrad(cublasHandle_t* handle, float* seed);
    void computeGrad(cublasHandle_t* handle);
    virtual void changeMemory(float* d_newMem);
    AD(std::string _name, unsigned int _rows, unsigned int _cols);
    AD();
    virtual ~AD();
};

struct AbstractCol : AD {
    AbstractCol(std::string _name, unsigned int _rows);
    AbstractCol();
};

struct Col : AbstractCol {
    void loadValues(std::valarray<float> newValues);
    Col(std::string _name, unsigned int _rows);
};

struct Matrix : AD {
    void loadValues(std::valarray<float> newValues);
    void gradDescent(cublasHandle_t* handle, float learningRate);
    Matrix(std::string _name, unsigned int _rows, unsigned int _cols);
};

struct Flatten : AbstractCol {
    AD* source;
    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    void changeMemory(float* d_newMem) override;
    Flatten(AD* source);
    ~Flatten() override;
};

struct MatrixColProduct : AbstractCol {
    AD* matrix;
    AD* col;
    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    MatrixColProduct(AD* m, AD* x);
    ~MatrixColProduct() override;
};

struct ColLeakyReLU : AD {
    AD* col;
    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    ColLeakyReLU(AD* _col);
    ~ColLeakyReLU() override;
};

struct Scalar : AD {
    AD* col;
    float scalar;
    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    Scalar(AD* _col, float _scalar);
    ~Scalar() override;
};

struct Add : AD {
    AD* col1;
    AD* col2;
    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    Add(AD* _col1, AD* _col2);
    ~Add() override;
};

struct InnerProduct : AbstractCol {
    AD* col1;
    AD* col2;
    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    InnerProduct(AD* _col1, AD* _col2);
    ~InnerProduct() override;
};

struct Convolution : AD {
    AD* multiplicand;
    AD* kernel;

    float* d_kernel;
    float* d_input;

    unsigned int rowPadding;
    unsigned int rowSkip;
    unsigned int colPadding;
    unsigned int colSkip;

    unsigned int unrKrnlRows;
    unsigned int unrKrnlCols;

    void unrollKernel();
    void padInput();

    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    Convolution(AD* multiplicand, AD* kernel, unsigned int rowPadding,
        unsigned int rowSkip, unsigned int colPadding,
        unsigned int colSkip);
    ~Convolution() override;
};

struct MaxPool : AD {
    AD* matrix;

    unsigned int rowSkip;
    unsigned int colSkip;

    unsigned int width;
    unsigned int height;

    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;

    MaxPool(AD* m, unsigned int width, unsigned int height, unsigned int rowSkip,
        unsigned int colSkip);
    ~MaxPool() override;
};

struct ConcatCol : AD {
    std::vector<AD*> cols;
    void resetGrad() override;
    void pushGrad(cublasHandle_t* handle, float* d_seed) override;
    void compute(cublasHandle_t* handle) override;
    ConcatCol(std::vector<AD*> cols);
    ~ConcatCol() override;
};

} // namespace FA

#define CUBLAS_CHECK(err)                                                        \
    do {                                                                         \
        cublasStatus_t err_ = (err);                                             \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                            \
        }                                                                        \
    } while (0)

#endif /* SILLY_AUTODIFF_H */
