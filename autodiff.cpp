#include <iostream>
#include<valarray>
#include<set>
#include<initializer_list>

using namespace std;

struct AD {
     virtual double operator()(initializer_list<double> args) = 0;
     virtual double operator()() = 0;
     virtual ostream& to_stream(ostream& os)  = 0;

     double val = 0;
     void setValue( double _val) {
         this->val = _val;
     }
     set<AD*> deps;


};

ostream& operator<<(ostream& os, AD& obj) {
    return obj.to_stream(os);
}


struct AD_Constant: AD {
  double val;
  AD_Constant(double val): val(val) {
      this->deps = {};
  }

  double operator()(initializer_list<double> args) { return this->val; }
  double operator()() { return this->val; }


  ostream& to_stream(ostream& os) {
      return os << " AD CONSTANT " << this->val << " " << this;
  }
};


struct AD_Plus: AD {
  AD &augend, &addend;

  AD_Plus(AD& _augend, AD& _addend): augend(_augend), addend(_addend) {
      this->deps = {};
      this->deps.insert(_augend.deps.begin(), _augend.deps.end());
      this->deps.insert(_addend.deps.begin(), _addend.deps.end());
  }

  double operator()() {
      return this->augend() + this->addend();
  }

  double operator()(initializer_list<double> args) {
      if(args.size() == 0) 
        return this->augend() + this->addend();

      int i {0};
      for(auto var : this->deps){ 
        var->setValue(*(data(args) + i));
        i++;

      }
      return this->augend() + this->addend();
  }

  ostream& to_stream(ostream& os ) {
      return os << " AD Plus "  << this;
  }

};

struct AD_Mul: AD {
    AD &plier, &plicand;

    AD_Mul(AD& _plier, AD& _plicand): plier(_plier), plicand(_plicand) {
      this->deps = {};
      this->deps.insert(_plicand.deps.begin(), _plicand.deps.end());
      this->deps.insert(_plier.deps.begin(), _plier.deps.end());
    }

    double operator()() {
        return this->plier() + this->plicand();
    }

    double operator()(initializer_list<double> args) {
      if(args.size() == 0) 
        return this->plier() * this->plicand();

      int i {0};
      for(auto *var : this->deps) {
        cout << *var << " : " << *(data(args) + i) << "\n";
        var->setValue(*(data(args) + i));
        i++;

      }

      return this->plier() * this->plicand();
    }
    ostream& to_stream(ostream& os) {
        return os << " AD Mul " << this;
    }
};


struct AD_Variable : AD {
  string name;
  AD_Variable (double _val, string _name): name(_name) {
      this->deps = {this};
      this->val = _val;
  }

  double operator()() {
      return this->val;
  }

  double operator()(initializer_list<double> args) {
      return this->val;
  }
  ostream& to_stream(ostream& os) {
      return os << " AD Var " << this->name << " " << this;
  }

};

AD_Plus operator+(AD& augend, AD& addend) {
    return AD_Plus(augend, addend);
}

AD_Mul operator*(AD& plier, AD& plicand) {
    return AD_Mul(plier, plicand);
}


int main() {
    AD_Variable x (0,"x");
    AD_Variable y (0,"y");

    AD_Plus d_ = x + y;
    AD_Variable z (0,"z");




    AD_Mul d = d_ * z ;
    cout << "hello, world!\n" << d({5.0,9.0, 1.0}) << "\n";

    return 0;
}
