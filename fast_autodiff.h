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
    float* grad;
    virtual void compute();
    virtual void resetGrad();
    virtual void pushGrad(float seed[]);
    AD(std::string _name, unsigned int _rows, unsigned int _cols);
} ;

struct AbstractCol : AD {
    float* value;
    AbstractCol(std::string _name, unsigned int _rows);
    ~AbstractCol();
};

struct Col : AbstractCol {
   void loadValues(std::valarray<float> newValues);
   Col(std::string _name, unsigned int _rows);
};

struct Matrix : AD {
    float* value;
    void loadValues(std::valarray<float> newValues);
    Matrix(std::string _name, unsigned int _rows, unsigned int _cols);
    ~Matrix();
};

struct MatrixColProduct : AbstractCol {
    Matrix* matrix;
    AbstractCol* col;
    void resetGrad() override;
    void pushGrad(float seed[]) override;
    void compute() override;
    MatrixColProduct(Matrix* m, AbstractCol* x);
};

struct ColLeakyReLU : AbstractCol {
    AbstractCol* col;
    void resetGrad() override;
    void pushGrad(float seed[]) override;
    void compute() override;
    ColLeakyReLU(AbstractCol* _col);
};

struct Scalar : AbstractCol {
    AbstractCol* col;
    float scalar;
    void resetGrad() override;
    void pushGrad(float seed[]) override;
    void compute() override;
    Scalar(AbstractCol* _col, float _scalar);
};

struct AddCol : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(float seed[]) override;
    void compute() override;
    AddCol(AbstractCol* _col1, AbstractCol* _col2);
};

struct InnerProduct : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;
    void resetGrad() override;
    void pushGrad(float seed[]) override;
    void compute() override;
    InnerProduct(AbstractCol* _col1, AbstractCol* _col2);
};

}


#endif /* FAST_AUTODIFF_H */
