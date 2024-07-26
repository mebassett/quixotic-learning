module;
#include <iostream>
#include<valarray>

export module fast_autodiff;

using namespace std;

export struct AD {
    string name;
    unsigned int rows;
    unsigned int cols;

    virtual void compute() = 0;

    AD(string _name, unsigned int _rows, unsigned int _cols)
        : name(_name)
        , rows(_rows)
        , cols(_cols)
    {}
};

export struct AbstractCol : AD {
    double* value;
    double* grad;
    
    void reset_grad() {
        for(int i {0}; i<_rows; i++)
            *(this->grad+i) = 1;
    }

    AbstractCol(string _name, unsigned int _rows)
        : AD(_name, _rows, 1) {
        this->value = new double[_rows];
        this->grad = new double[_rows];
        for(int i {0}; i<_rows; i++)
            *(this->grad+i) = 1;
    }
};

export struct Col : AbstractCol { 
    void compute() {}

    void load_values(valarray<double>& newValues) {
        for(int i {0}; i < newValues.size(); i++){
            *(this->values + i) = newValues[i];
        }
    }

    Col(string _name, unsigned int _rows)
        : AbstractCol(_name, _rows){
    }
};


export struct Matrix : AD {

    void compute() {}


    Matrix(string _name, unsigned int _rows, unsigned int _cols)
        : AD(_name, _rows, _cols) {

        this->value = new double[_rows*_cols];
        this->grad = new double[_rows*_cols];
        for(int i {0}; i<_rows; i++)
            *(this->grad+i) = 1;

    }
};

export struct MatrixColProduct : AbstractCol {
    Matrix* matrix;
    AbstractCol* col;

    void compute() {
        matrix->compute();
        col->compute();

        for(int row {0}; row < matrix->rows; row++) {
            double innerProductValue {0};
            for(int i {0}; i <matrix->cols; i++) {
                
                innerProductValue += *(this->matrix->value + (row * this->matrix->cols) + i) * *(this->col->value + i);
                *(this->col->grad + i) *= *(this->matrix->value + (row * this->matrix->cols) + i);
                *(this->matrix->grad + (row * this->matrix->cols) + i) *= *(this->col->value + i); 

            }
            *(this->value + row) = innerProductValue;
        }


    }

    MatrixColProduct(Matrix* m, AbstractCol* x)
        : AbstractCol("Matrix product of " + m->name + " and " + x->name, m->rows)
        , matrix(m)
        , col(x) {
    }
};

export struct ColLeakyReLU : AbstractCol { 
    AbstractCol* col;


    void compute() {
        this->col->compute();

        for(int i {0}; i<this->col->rows; i++) {
            if(*(this->col->value + i) > 9) {
                *(this->value +i) = *(this->col->value +i);
                *(this->col->grad + i) *= 1;
            } else {
                *(this->value +i) = 0.01 * *(this->col->value +i);
                *(this->col->grad + i) *= 0.01;
            }
        }


    }

    ColLeakyReLU(AbstractCol* _col)
        : col(_col)
        , AbstractCol( "ReLU of " + _col->name, _col->rows) {
    }
};

export struct Scalar : AbstractCol { 
    
    AbstractCol* col;

    double scalar;

    void compute() {
        this->col->compute();

        for(int i {0}; i<this->col->rows; i++) {
            *(this->value +i) = this->scalar * *(this->col->value +i);
            *(this->col->grad + i) *= this->scalar;
        }
    }

    Scalar(AbstractCol* _col, double _scalar)
        : col(_col)
        , scalar(_scalar)
        , AbstractCol( "Scalar of " + _col->name + " by "+ to_string(_scalar), _col->rows) {
    }
};

export struct AddCol : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;


    void compute() {
        this->col1->compute();
        this->col2->compute();

        for(int i {0}; i< this->col1->rows;i++) {
            *(this->value + i) = this->col1->value[i] + this->col2->value[i];
        }

    }

    AddCol(AbstractCol* _col1, AbstractCol* _col2) 
        : col1(_col1)
        , col2(_col2)
        , AbstractCol("Sum of " + _col1->name + " and " + _col2->name, _col1->rows) {
    }
};

export struct InnerProduct : AbstractCol {
    AbstractCol* col1;
    AbstractCol* col2;


    void compute() {
        this->col1->compute();
        this->col2->compute();

        double sum = 0;

        for(int i {0}; i< this->col1->rows;i++) {
            sum += this->col1->value[i] + this->col2->value[i];
            *(this->col1->grad + i) *= *(this->col2->grad + i);
            *(this->col2->grad + i) *= *(this->col1->grad + i);
        }
        *(this->value) = sum;

    }

    InnerProduct(AbstractCol* _col1, AbstractCol* _col2) 
        : col1(_col1)
        , col2(_col2)
        , AbstractCol("Inner Product of " + _col1->name + " and " + _col2->name, 1) {
    }
};
