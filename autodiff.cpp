#include <iostream>
#include<valarray>
#include<set>
#include<cstdarg>

using namespace std;

struct AD {
     virtual double operator()(...) = 0;

     double val = 0;
     void setValue( double _val) {
         this->val = _val;
     }
     set<AD*> deps;
};

struct AD_Constant: AD {
  double val;
  AD_Constant(double val): val(val) {
      this->deps = {};
  }

  double operator()(...) { return this->val; }
};

struct AD_Plus: AD {
  AD &augend, &addend;

  AD_Plus(AD& _augend, AD& _addend): augend(_augend), addend(_addend) {
      this->deps = {};
      this->deps.insert(_augend.deps.begin(), _augend.deps.end());
      this->deps.insert(_addend.deps.begin(), _addend.deps.end());
  }


  double operator()(...) {
      va_list args;
      va_start(args, 0);
      cout << "length : " << this->deps.size() << "\n";
      for(auto var : this->deps){
        double val = va_arg(args, double);
        var->setValue(val);
      };
      va_end(args);
      return this->augend() + this->addend();
  }




};

struct AD_Mul: AD {
    AD &plier, &plicand;

    AD_Mul(AD& _plier, AD& _plicand): plier(_plier), plicand(_plicand) {
      this->deps = {};
      this->deps.insert(_plicand.deps.begin(), _plicand.deps.end());
      this->deps.insert(_plier.deps.begin(), _plier.deps.end());
    }

    double operator()(...) {
      va_list args;
      va_start(args, 0);
      cout << "length : " << this->deps.size() << "\n";
      for(auto var : this->deps){
        double val = va_arg(args, double);
        var->setValue(val);
      };
      va_end(args);
      return this->plier() * this->plicand();
    }
};


struct AD_Variable : AD {
  AD_Variable (double _val) {
      this->deps = {this};
      this->val = _val;
  }


  double operator()(...) {
      return this->val;
  }

};

AD_Plus operator+(AD& augend, AD& addend) {
    return AD_Plus(augend, addend);
}

AD_Mul operator*(AD& plier, AD& plicand) {
    return AD_Mul(plier, plicand);
}

int main() {
    AD_Variable x (2);
    AD_Variable y (3);

    AD_Plus d_ = x + y;
    AD_Variable z (5);




    AD_Mul d = d_ * z ;

    cout << "hello, world!\n" << d(5.0,9.0,2) << "\n";
    return 0;
}
