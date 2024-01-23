#include <iostream>
#include<valarray>

using namespace std;

struct AD {
     virtual double operator()() = 0;
};

struct AD_Constant: AD {
  double val;
  AD_Constant(double val): val(val) {}

  double operator()() { return this->val; }
};

struct AD_Plus: AD {
  AD &augend, &addend;

  AD_Plus(AD& _augend, AD& _addend): augend(_augend), addend(_addend) {}

  double operator()() {
    return this->augend() + this->addend();
  }




};

struct AD_Mul: AD {
    AD &plier, &plicand;

    AD_Mul(AD& _plier, AD& _plicand): plier(_plier), plicand(_plicand) {}

    double operator()() {
        return this->plier() * this->plicand();
    }
};


struct AD_Variable : AD {
  double val = 0;
  AD_Variable (double _val): val(_val) {}

  void setValue(double _val) {
      this->val = _val;
  }

  double operator()() {
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

    cout << "hello, world!\n" << d() << "\n";
    return 0;
}
