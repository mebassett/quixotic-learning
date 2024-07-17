module;
#include <iostream>

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

    AbstractCol(string _name, unsigned int _rows)
        : AD(_name, _rows, 1) {
        this->value = new double[_rows];
        this->grad = new double(_rows);
        for(int i {0}; i<_rows; i++)
            *(this->grad+i) = 1;
    }
};

export struct Col : AbstractCol { 
    void compute() {}

    Col(string _name, unsigned int _rows)
        : AbstractCol(_name, _rows){
        this->value = new double[_rows];
        this->grad = new double(_rows);
        for(int i {0}; i<_rows; i++)
            *(this->grad+i) = 1;
    }
};


export struct Matrix : AD {
    double* value;
    double* grad;

    void compute() {}

    Matrix(string _name, unsigned int _rows, unsigned int _cols)
        : AD(_name, _rows, _cols) {

        this->value = new double[_rows*_cols];
        this->grad = new double(_rows*_cols);
        for(int i {0}; i<_rows; i++)
            *(this->grad+i) = 1;

    }
};

export struct MatrixColProduct : AbstractCol {
    Matrix* matrix;
    AbstractCol* col;
    double* value;
    double* grad;

    void compute() {
        matrix->compute();
        col->compute();

        for(int row {0}; row < matrix->rows; row++) {
            double innerProductValue {0};
            for(int i {0}; i <matrix->cols; i++) {
                
                innerProductValue += *(this->matrix->value + (row * this->matrix->cols) + i) * *(this->col->value + i);
                *(this->col->value + i) *= *(this->matrix->value + (row * this->matrix->cols) + i);
                *(this->matrix->grad + (row * this->matrix->cols) + i) *= *(this->col->value + i); 

            }
            *(this->value + row) = innerProductValue;
        }


    }

    MatrixColProduct(Matrix* m, AbstractCol* x)
        : AbstractCol("Matrix product of " + m->name + " and " + x->name, matrix->rows)
        , matrix(m)
        , col(x) {
        this->value = new double[matrix->rows];
        this->grad = new double(matrix->rows);
        for(int i {0}; i<matrix->rows; i++)
            *(this->grad+i) = 1;
    }
};

int main() {
    cout << "test \n";
}
