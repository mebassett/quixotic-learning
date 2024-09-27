// fast_autodiff.h
#ifndef FAST_AUTODIFF_H
#define FAST_AUTODIFF_H
#include<valarray>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace FA {

struct AD {
    std::string name;
    unsigned int rows;
    unsigned int cols;
    float* d_grad;
    float* d_value;
    float* value;
    virtual void fromDevice();
    virtual void compute(cublasHandle_t *handle);
    virtual void resetGrad();
    virtual void pushGrad(cublasHandle_t *handle, float* seed);
    void computeGrad(cublasHandle_t *handle );
    AD(std::string _name, unsigned int _rows, unsigned int _cols);
    virtual ~AD();
} ;


struct AbstractCol : AD {
    AbstractCol(std::string _name, unsigned int _rows);
};

struct Col : AbstractCol {
   void loadValues(std::valarray<float> newValues);
   Col(std::string _name, unsigned int _rows);
};

struct Matrix : AD {
    void loadValues(std::valarray<float> newValues);
    void gradDescent(cublasHandle_t *handle, float learningRate);
    Matrix(std::string _name, unsigned int _rows, unsigned int _cols);
};

struct MatrixColProduct : AbstractCol {
    AD* matrix;
    AbstractCol* col;
    void resetGrad() override;
    void pushGrad(cublasHandle_t *handle, float* d_seed) override;
    void compute(cublasHandle_t *handle) override;
    MatrixColProduct(AD* m, AbstractCol* x);
    ~MatrixColProduct() override;
};

struct ColLeakyReLU : AbstractCol {
    AbstractCol* col;
    void resetGrad() override;
    void pushGrad(cublasHandle_t *handle, float* d_seed) override;
    void compute(cublasHandle_t *handle) override;
    ColLeakyReLU(AbstractCol* _col);
    ~ColLeakyReLU() override;

};

struct Scalar : AbstractCol {
    AbstractCol* col;
    float scalar;
    void resetGrad() override;
    void pushGrad(cublasHandle_t *handle, float* d_seed) override;
    void compute(cublasHandle_t *handle) override;
    Scalar(AbstractCol* _col, float _scalar);
    ~Scalar() override;
};

struct AddCol : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(cublasHandle_t *handle, float* d_seed) override;
    void compute(cublasHandle_t *handle) override;
    AddCol(AbstractCol* _col1, AbstractCol* _col2);
    ~AddCol() override;
};

struct InnerProduct : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(cublasHandle_t *handle, float* d_seed) override;
    void compute(cublasHandle_t *handle) override;
    InnerProduct(AbstractCol* _col1, AbstractCol* _col2);
    ~InnerProduct() override;
};

struct Convolution : AD {
    Matrix* multiplicand;
    float* d_kernel;
    unsigned int kernelRows;
    unsigned int kernelCols;
    unsigned int rowPadding;
    unsigned int rowSkip;
    unsigned int colPadding;
    unsigned int colSkip;

    void resetGrad() override;
    void compute(cublasHandle_t *handle);
    void loadKernel(std::valarray<float> newKernel);
    Convolution(Matrix* m, 
                unsigned int kRows, unsigned int kCols,
                unsigned int rowPadding, unsigned int rowSkip,
                unsigned int colPadding, unsigned int colSkip);
    ~Convolution() override;

};

}

#define CUBLAS_CHECK(err) \
    do { \
        cublasStatus_t err_ = (err); \
        if (err_ != CUBLAS_STATUS_SUCCESS) { \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error"); \
        } \
    } while (0)


#endif /* FAST_AUTODIFF_H */
