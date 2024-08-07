// fast_autodiff.h
#ifndef FAST_AUTODIFF_H
#define FAST_AUTODIFF_H
#include<valarray>
#include <iostream>

namespace FA {

struct AD {
    std::string name;
    unsigned int rows;
    unsigned int cols;
    float* d_grad;
    virtual void compute();
    virtual void resetGrad();
    virtual void pushGrad(float* seed);
    AD(std::string _name, unsigned int _rows, unsigned int _cols);
} ;

struct AbstractCol : AD {
    float* d_value;
    AbstractCol(std::string _name, unsigned int _rows);
    virtual ~AbstractCol();
};

struct Col : AbstractCol {
   void loadValues(std::valarray<float> newValues);
   Col(std::string _name, unsigned int _rows);
};

struct Matrix : AD {
    float* value;
    float* d_value;
    void loadValues(std::valarray<float> newValues);
    void gradDescent(float learningRate);
    Matrix(std::string _name, unsigned int _rows, unsigned int _cols);
    ~Matrix();
};

struct MatrixColProduct : AbstractCol {
    Matrix* matrix;
    AbstractCol* col;
    float* value;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    void fromDevice();
    MatrixColProduct(Matrix* m, AbstractCol* x);
    ~MatrixColProduct();
};

struct ColLeakyReLU : AbstractCol {
    AbstractCol* col;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    ColLeakyReLU(AbstractCol* _col);
};

struct Scalar : AbstractCol {
    AbstractCol* col;
    float* value;
    float scalar;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    void fromDevice();
    void computeGrad();
    Scalar(AbstractCol* _col, float _scalar);
    ~Scalar();
};

struct AddCol : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    AddCol(AbstractCol* _col1, AbstractCol* _col2);
};

struct InnerProduct : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    InnerProduct(AbstractCol* _col1, AbstractCol* _col2);
};

}


#endif /* FAST_AUTODIFF_H */
