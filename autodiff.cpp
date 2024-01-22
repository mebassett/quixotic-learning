#include <iostream>
#include<valarray>

using namespace std;

struct AD {
     virtual double operator()(valarray<double> x) = 0 ;
};

struct AD_Constant: AD {
  double val;
  AD_Constant(double val): val(val) {}

  double operator()(valarray<double> x) { return this->val; }
};

struct AD_Plus: AD {
  AD &augend, &addend;

  AD_Plus(AD& _augend, AD& _addend): augend(_augend), addend(_addend) {}

  double operator()(valarray<double> x) {
    if(x.size() != 2) 
        throw invalid_argument("Arity does not match in Plus.");
    return this->augend(x[slice(0,1,0)]) + this->addend(x[slice(1,1,0)]);
  }
};

struct AD_Variable : AD {

  double operator()(valarray<double> x) {
      if(x.size() != 1)
          throw invalid_argument("Arity does not match.");


      return x[0];
  }

  AD_Plus operator+(AD& addend) {
      return AD_Plus(*this, addend);
  }

};

int main() {
    AD_Variable x ;
    AD_Variable y ;


    AD_Plus z = y + x;

    cout << "hello, world!\n" << z({2,3}) << "\n";
    return 0;
}
