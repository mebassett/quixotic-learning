#include <iostream>
#include<valarray>
#include<map>

using namespace std;

struct ADV {
    virtual void take_gradient(valarray<double> seed) = 0;
    virtual valarray<double>& get_gradient() = 0;
    virtual valarray<double>& operator()(map<string, valarray<double>> args) = 0;
    virtual valarray<double>& operator()() = 0;
    virtual void setValue (valarray<double> _val) = 0;
    virtual const unsigned int size() = 0;
    virtual ostream& to_stream(ostream& os) = 0;
    map<string, ADV*> deps;
    valarray<double> val;
    string name;
};

ostream& operator<<(ostream& os, ADV& obj) {
    return obj.to_stream(os);
}

struct ADV_Vec : ADV {
    valarray<double> grad;
    unsigned int vector_length;

    void take_gradient(valarray<double> seed) {
       this->grad += seed; 
        
    }
    valarray<double>& get_gradient() {
        return this->grad;
    }
    valarray<double>& operator()() {
        return this->val;
    }
    valarray<double>& operator()(map<string, valarray<double>> args) {
        return (*this)();
    }
    void setValue (valarray<double> _val)  {
        this->val = _val;
    }

    const unsigned int size() {
        return this->val.size();
    }

    ostream& to_stream(ostream& os) {
      return os << " AD Vector " << this->name << " @ " << this;
    }

    ADV_Vec(string _name, unsigned int _size) : vector_length(_size) {
        this->val = valarray<double>(_size);
        this->grad  =valarray<double>( (double) 0 , _size);
        this->name = _name;
        this->deps = {{_name, this}};
    }

    ~ADV_Vec() {
    }

};

struct ADV_InnerProduct: ADV {
    valarray<double> grad;
    ADV& vec1;
    ADV& vec2;

    unsigned int input_size;
    void take_gradient(valarray<double> seed) {

        //seed.size() == this->size(), id est 1
        this->vec1.take_gradient(seed[0] * this->vec2());
        this->vec2.take_gradient(seed[0] * this->vec1());
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        valarray<double> v1 = this->vec1(); 
        valarray<double> v2 = this->vec2(); 

        this->val = { (v1 + v2).sum() };
        return this->val;
        
    }
    valarray<double>& operator()(map<string, valarray<double>> args) {
        for(auto [vector_name, value] : args)
            this->deps.at(vector_name)->setValue(value);

        return (*this)();
    }
    void setValue(valarray<double> _val) {}

    const unsigned int size(){
        return 1;
    }

    ostream& to_stream(ostream& os) {
        return os << " ADV InnerProduct of " << this->vec1.name << " @ " 
                  << &(this->vec1) << " and " << this->vec2.name << " @ "
                  << &(this->vec2) << "\n";
    }

    ADV_InnerProduct(ADV& _v1, ADV& _v2): vec1(_v1), vec2(_v2) {
      //need to check that their sizes are equal.
      if(_v1.size() != _v2.size()) throw out_of_range("ADVInnerProduct: vectors are not the same size.");
      this->deps = {};
      this->deps.merge(_v1.deps);
      this->deps.merge(_v2.deps);
      this->name = "ADVInnerProduct of " + _v1.name + " and " + _v2.name;
    }


};

// need: 
// ADV_VectorSum (must be same size)
// ADV_VectorProduct (must be same size, or one must be size 1)
// ADV_exp (acts on every component), also ADV_sin, ADV_cos, maybe ADV_ReLU
// ADV_Concat (takes a list of size 1s and gives you a vec..)
// I think that should do it!
//

struct ADV_Sum : ADV {
    valarray<double> grad;
    valarray<double> value;
    ADV& vec1;
    ADV& vec2;
    void take_gradient(valarray<double> seed) {
        this->grad += seed;
        this->vec1.take_gradient(seed);
        this->vec2.take_gradient(seed);
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        this->val = this->vec1() + this->vec2();
        return this->val;
    }

    valarray<double>& operator()(map<string, valarray<double>> args) {
        for(auto [name, value] : args)
            this->deps.at(name)->setValue(value);
        return (*this)();
    }

    void setValue(valarray<double> _val) {}

    const unsigned int size() {
        return this->vec1.size();
    }

    ostream& to_stream(ostream& os) {
        return os << this->name;
    }

    ADV_Sum(ADV& _v1, ADV& _v2): vec1(_v1), vec2(_v2) {
        if(_v1.size() != _v2.size()) 
            throw out_of_range("ADVSum: vectors are not the same size");
        this->deps = {};
        this->deps.merge(_v1.deps);
        this->deps.merge(_v2.deps);
        this->name = "ADVSum of " + _v1.name + " and " + _v2.name;
        this->grad = valarray<double>((double)0,_v1.size());

    }
};



struct AD {
     virtual double operator()(map<string, double> args) = 0;
     virtual double operator()() = 0;
     virtual void take_gradient(double seed) = 0;
     virtual double get_gradient() = 0;
     virtual ostream& to_stream(ostream& os)  = 0;

     double val = 0;
     void setValue( double _val) {
         this->val = _val;
     }
     map<string, AD*> deps;


};

ostream& operator<<(ostream& os, AD& obj) {
    return obj.to_stream(os);
}


struct AD_Constant: AD {
  double val;
  AD_Constant(double val): val(val) {
      this->deps = {};
  }

  double operator()(map<string, double> args) { return this->val; }
  double operator()() { return this->val; }

  void take_gradient(double seed) {}

  double get_gradient() {return 0;}


  ostream& to_stream(ostream& os) {
      return os << " AD CONSTANT " << this->val << " " << this;
  }
};


struct AD_Plus: AD {
  AD &augend, &addend;

  AD_Plus(AD& _augend, AD& _addend): augend(_augend), addend(_addend) {
      this->deps = {};
      this->deps.merge(_augend.deps);
      this->deps.merge(_addend.deps);
  }

  double operator()() {
      this->val = this->augend() + this->addend();
      return this->val;
  }

  double operator()(map<string, double> args) {
      if(args.size() == 0) 
        return this->augend() + this->addend();

      for(const auto& [varname, value] : args)
        (this->deps.at(varname))->setValue(value);


      this->val = this->augend() + this->addend();
      return this->val;
  }
  ostream& to_stream(ostream& os ) {
      return os << " AD Plus "  << this;
  }

  void take_gradient(double seed) { 
      this->addend.take_gradient(seed);
      this->augend.take_gradient(seed);
  }

  map<string, double> grad() {
      map<string, double> m;
        this->take_gradient(1.0);
      for(const auto& [varname, var] : this->deps) {
          m.insert({varname, var->get_gradient()});
      }
      return m;
  }
  double get_gradient() {return 0;}



};

struct AD_Mul: AD {
    AD &plier, &plicand;

    AD_Mul(AD& _plier, AD& _plicand): plier(_plier), plicand(_plicand) {
      this->deps = {};
      this->deps.merge(_plicand.deps);
      this->deps.merge(_plier.deps);
    }

    double operator()() {
        this->val = this->plier() * this->plicand();
        return val;
    }

    double operator()(map<string, double> args) {
      if(args.size() == 0) 
        return this->plier() * this->plicand();

      for(const auto& [varname, value] : args)
        (this->deps.at(varname))->setValue(value);


      this->val = this->plier() * this->plicand();
      return val;
    }
    ostream& to_stream(ostream& os) {
        return os << " AD Mul " << this;
    }
    void take_gradient(double seed) { 
        this->plier.take_gradient(seed * this->plicand.val);
        this->plicand.take_gradient(seed * this->plier.val);
    }

    map<string, double> grad() {
        map<string, double> m;

        this->take_gradient(1.0);

        for(const auto& [varname, var] : this->deps) {
            m.insert({varname, var->get_gradient()});
        }
        return m;
    }
  double get_gradient() {return 0;}

};


struct AD_Variable : AD {
  string name;
  double partial;
  AD_Variable (string _name): name(_name) {
      this->deps = {{_name, this}};
  }

  double operator()() {
      return this->val;
  }

  double operator()(map<string, double> args) {
      return this->val;
  }
  ostream& to_stream(ostream& os) {
      return os << " AD Var " << this->name << " " << this;
  }

  void take_gradient(double seed) {
      this->partial += seed;
  }
  double get_gradient(){
      return this->partial;
  }
};

AD_Plus operator+(AD& augend, AD& addend) {
    return AD_Plus(augend, addend);
}

AD_Mul operator*(AD& plier, AD& plicand) {
    return AD_Mul(plier, plicand);
}


int main() {
    //AD_Variable x ("x");
    //AD_Variable y ("y");

    //AD_Plus d_ = x + y;
    //AD_Variable z ("z");




    //AD_Mul d = d_ * z ;
    //cout << "hello, world!\n" << d({{"x", 5.0}, {"y", 9.0}, {"z", 2.0}}) << "\n";

    //map<string, double> grad = d.grad();

    //cout << "x partial " << x.get_gradient() << "\n";

    //for(const auto& [varname, partial] : grad) 
    //    cout << "(del f/del " << varname << ") = " << partial << "\n";
    //
    ADV_Vec x ("x",3);

    ADV_Vec y {"y",3};

    ADV_InnerProduct z (x, x);

    double inner_product = z({ {"x", {1,2,3}}})[0];

    cout << z << " : " << inner_product << "\n";


    z.take_gradient({1});

    valarray<double> x_grad = x.get_gradient();
    cout << "x0: " << x_grad[0] << " x1: " << x_grad[1] << " x2: " << x_grad[2] <<"\n";

    return 0;
}
