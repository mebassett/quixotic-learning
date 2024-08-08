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
    float* d_value;
    float* value;
    virtual void fromDevice();
    virtual void compute();
    virtual void resetGrad();
    virtual void pushGrad(float* seed);
    void computeGrad();
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
    void gradDescent(float learningRate);
    Matrix(std::string _name, unsigned int _rows, unsigned int _cols);
};

struct MatrixColProduct : AbstractCol {
    Matrix* matrix;
    AbstractCol* col;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    MatrixColProduct(Matrix* m, AbstractCol* x);
    ~MatrixColProduct() override;
};

struct ColLeakyReLU : AbstractCol {
    AbstractCol* col;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    ColLeakyReLU(AbstractCol* _col);
    ~ColLeakyReLU() override;

};

struct Scalar : AbstractCol {
    AbstractCol* col;
    float scalar;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    Scalar(AbstractCol* _col, float _scalar);
    ~Scalar() override;
};

struct AddCol : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    AddCol(AbstractCol* _col1, AbstractCol* _col2);
    ~AddCol() override;
};

struct InnerProduct : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(float* d_seed) override;
    void compute() override;
    InnerProduct(AbstractCol* _col1, AbstractCol* _col2);
    ~InnerProduct() override;
};

}


#endif /* FAST_AUTODIFF_H */
