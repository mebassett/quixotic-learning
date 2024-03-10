module;
#include <iostream>
#include<valarray>
#include<vector>
#include<map>

export module autodiff;
using namespace std;


export struct ADV {
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

export ostream& operator<<(ostream& os, ADV& obj) {
    return obj.to_stream(os);
}

export struct ADV_Vec : ADV {
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

export struct ADV_InnerProduct: ADV {
    valarray<double> grad;
    ADV* vec1;
    ADV* vec2;

    unsigned int input_size;
    void take_gradient(valarray<double> seed) {

        this->vec1->take_gradient(seed * this->vec2->val);
        this->vec2->take_gradient(seed * this->vec1->val);
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        valarray<double> v1 = (*this->vec1)(); 
        valarray<double> v2 = (*this->vec2)(); 

        this->val = { (v1 * v2).sum() };
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
        return os << " ADV InnerProduct of " << this->vec1->name << " @ " 
                  << &(this->vec1) << " and " << this->vec2->name << " @ "
                  << &(this->vec2) << "\n";
    }

    ADV_InnerProduct(ADV* _v1, ADV* _v2): vec1(_v1), vec2(_v2) {
      this->name = "ADVInnerProduct of " + _v1->name + " and " + _v2->name;
      //need to check that their sizes are equal.
      if(_v1->size() != _v2->size()) throw out_of_range("ADVInnerProduct ("+ this->name +"): vectors are not the same size.");
      this->deps = {};
      this->deps.merge(_v1->deps);
      this->deps.merge(_v2->deps);
    }


};

// need: 
// ADV_VectorSum (must be same size)
// ADV_VectorProduct (must be same size, or one must be size 1)
// ADV_exp (acts on every component), also ADV_sin, ADV_cos, maybe ADV_ReLU
// ADV_Concat (takes a list of size 1s and gives you a vec..)
// I think that should do it!
//

export struct ADV_Sum : ADV {
    valarray<double> grad;
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

export struct ADV_VectorProduct : ADV {
    valarray<double> grad;
    ADV& vec1;
    ADV& vec2;
    void take_gradient(valarray<double> seed) {
        this->grad += seed;
        this->vec1.take_gradient(seed * this->vec2());
        this->vec2.take_gradient(seed * this->vec1());
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        this->val = this->vec1() * this->vec2();
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

    ADV_VectorProduct(ADV& _v1, ADV& _v2): vec1(_v1), vec2(_v2) {
        if(_v1.size() != _v2.size()) 
            throw out_of_range("ADVSum: vectors are not the same size");
        this->deps = {};
        this->deps.merge(_v1.deps);
        this->deps.merge(_v2.deps);
        this->name = "ADVSum of " + _v1.name + " and " + _v2.name;
        this->grad = valarray<double>((double)0,_v1.size());

    }
};

export struct ADV_Concat : ADV {
    vector<ADV*> members;
    valarray<double> grad;

    void take_gradient(valarray<double> seed) {
        // seed.size == sum(member => member.sizer() for member in members
        // we will assume member.size() == 1 always
        for(int i {0}; i<this->members.size(); i++) {
            this->members[i]->take_gradient(seed[slice(i,1,1)]);
        }
        this->grad = seed;
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        this->val = valarray<double>(this->members.size());
        for(int i {0}; i<this->members.size();i++){
            this->val[i] = (*this->members[i])()[0];
        }
        return this->val;
    }
    valarray<double>& operator()(map<string, valarray<double>> args) {
        for(auto [name, value] : args)
            this->deps.at(name)->setValue(value);
        return (*this)();
    }
    void setValue(valarray<double> _val) {}
    const unsigned int size() {
        return this->members.size();
    }
    ostream& to_stream(ostream& os) {
        return os << this->name;
    }

    ADV_Concat(vector<ADV*> _members): members(_members) {
        // should have a check to ensure they are all size 1...
        this->deps = {};
        this->name = "ADVConcat of ";
        for (auto m : _members) {
            this->deps.merge(m->deps);
            this->name += m->name + " ";
        }
        this->grad = valarray<double>((double)0,_members.size());
    }
};

export struct ADV_Exp : ADV {
    ADV* input;
    valarray<double> grad;

    void take_gradient(valarray<double> seed) {
        this->input->take_gradient(this->val * seed) ;
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        this->val= exp((*this->input)());
        return this->val;
    }
    valarray<double>& operator()(map<string, valarray<double>> args) {
        for(auto [name, value] : args)
            this->deps.at(name)->setValue(value);
        return (*this)();
    }
    void setValue(valarray<double> _val) {}
    const unsigned int size() {
        return this->input->size();
    }
    ostream& to_stream(ostream& os) {
        return os << this->name;
    }

    ADV_Exp(ADV* _input): input(_input) {
        // should have a check to ensure they are all size 1...
        this->deps = _input->deps;
        this->name = "ADVExp of " + _input->name;
        this->grad = valarray<double>((double)0,_input->size());
    }
};

export struct ADV_Negate : ADV {
    ADV* input;
    valarray<double> grad;

    void take_gradient(valarray<double> seed) {
        this->input->take_gradient(-1 * seed) ;
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        this->val = - (*this->input)();
        return this->val;
    }
    valarray<double>& operator()(map<string, valarray<double>> args) {
        for(auto [name, value] : args)
            this->deps.at(name)->setValue(value);
        return (*this)();
    }
    void setValue(valarray<double> _val) {}
    const unsigned int size() {
        return this->input->size();
    }
    ostream& to_stream(ostream& os) {
        return os << this->name;
    }

    ADV_Negate(ADV* _input): input(_input) {
        // should have a check to ensure they are all size 1...
        this->deps = _input->deps;
        this->name = "ADVNegate of " + _input->name;
        this->grad = valarray<double>((double)0,_input->size());
    }
};

export struct ADV_LeakyReLU : ADV {
    ADV* input;
    valarray<double> grad;

    void take_gradient(valarray<double> seed) {
        this->input->take_gradient(this->grad * seed) ;
    }

    valarray<double>& get_gradient() {
        return this->grad;
    }

    valarray<double>& operator()() {
        valarray<double> ret = (*this->input)();
        for(int i {0}; i< ret.size(); i++) 
            if(ret[i] <= 0) {
                this->val[i] = ret[i] * 0.001;
                this->grad[i] = 0.001;
            } else {
                this->val[i] = ret[i];
                this->grad[i] = 1;
            }

        return this->val;
    }
    valarray<double>& operator()(map<string, valarray<double>> args) {
        for(auto [name, value] : args)
            this->deps.at(name)->setValue(value);
        return (*this)();
    }
    void setValue(valarray<double> _val) {}
    const unsigned int size() {
        return this->input->size();
    }
    ostream& to_stream(ostream& os) {
        return os << this->name;
    }

    ADV_LeakyReLU(ADV* _input): input(_input) {
        // should have a check to ensure they are all size 1...
        this->deps = _input->deps;
        this->name = "ADVLeakyReLU of " + _input->name;
        this->grad = valarray<double>((double)0,_input->size());
        this->val = valarray<double>((double)0,_input->size());
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


//int main() {
//    //AD_Variable x ("x");
//    //AD_Variable y ("y");
//
//    //AD_Plus d_ = x + y;
//    //AD_Variable z ("z");
//
//
//
//
//    //AD_Mul d = d_ * z ;
//    //cout << "hello, world!\n" << d({{"x", 5.0}, {"y", 9.0}, {"z", 2.0}}) << "\n";
//
//    //map<string, double> grad = d.grad();
//
//    //cout << "x partial " << x.get_gradient() << "\n";
//
//    //for(const auto& [varname, partial] : grad) 
//    //    cout << "(del f/del " << varname << ") = " << partial << "\n";
//    //
//    //ADV_Vec x ("x",3);
//
//    //ADV_Vec y {"y",3};
//
//    //ADV_InnerProduct z (x, x);
//
//    //double inner_product = z({ {"x", {1,2,3}}})[0];
//
//    //cout << z << " : " << inner_product << "\n";
//
//
//    //z.take_gradient({1});
//
//    //valarray<double> x_grad = x.get_gradient();
//    //cout << "x0: " << x_grad[0] << " x1: " << x_grad[1] << " x2: " << x_grad[2] <<"\n";
//    //
//
//    //ADV_Vec x ("x",1);
//    //ADV_Vec y ("y",1);
//    //ADV_Vec z ("z",2);
//
//    //ADV_Concat xy ({&x, &y});
//    //ADV_InnerProduct in (xy,z);
//
//    //double inner_product = in({ {"x",{3}}, {"y", {5}}, {"z",{7,9}}})[0];
//
//    //cout << in << " : " << inner_product << "\n";
//    //in.take_gradient({1});
//    //valarray<double> x_grad = z.get_gradient();
//
//    //cout << "del in / del x : " << x_grad[1] << "\n";
//
//    //ADV_Vec x ("x", 3);
//    //ADV_Vec y ("y", 3);
//    //ADV_Sum xy (x, y);
//    //ADV_Exp e (&xy);
//    //ADV_VectorProduct xexp (x, e);
//    //ADV_InnerProduct xexp2 (xexp, xexp);
//
//    //ADV_Vec z ("z", 1);
//    //ADV_Exp ze (&z);
//
//    //ADV_Concat cc ({ &ze, &xexp2 });
//    //ADV_InnerProduct op (cc, cc);
//
//    //valarray<double> result = op({ {"x", {0,1,2}}, {"y", {3,4,5}}, {"z", {117}}} );
//    //op.take_gradient({1,1,1});
//    //valarray<double> grad = z.get_gradient();
//
//    //for(auto i : result){
//    //    cout << "result " << i << "\n";
//    //}
//    //for(auto i : grad){
//    //    cout << "grad " << i << "\n";
//    //}
//    //ADV_Vec x ("x",1);
//    //ADV_Vec y ("y",1);
//    //ADV_VectorProduct y2 (y, y);
//
//    //ADV_Sum xy (x, y2);
//    //ADV_Exp e (&x);
//
//    //ADV_VectorProduct op (xy, e);
//
//    //double result = op( { {"x", {0}}, {"y", {2}}})[0];
//
//    //op.take_gradient({1});
//
//    //double x_grad = x.get_gradient()[0];
//    //double y_grad = y.get_gradient()[0];
//
//    //cout << "del_x f (0,1): " << x_grad << "\n";
//    //cout << "del_y f (0,1): " << y_grad << "\n";
//
//    //ADV_Vec x ("x", 1);
//    //ADV_Vec y ("y", 1);
//    //ADV_InnerProduct in (x, y);
//    //ADV_LeakyReLU op (&in);
//
//    //double result = op ({ {"x", {2}}, {"y", {-1}}})[0];
//
//    //cout << "f(x,y) = ReLU(xy), f(2,-1): " << result << "\n";
//    //op.take_gradient({1});
//
//    //double x_grad = x.get_gradient()[0];
//    //double y_grad = y.get_gradient()[0];
//
//    //cout << "del_x f (2,-1): " << x_grad << "\n";
//    //cout << "del_y f (2,-1)): " << y_grad << "\n";
//    //
//    ADV_Vec x ("x", 2);
//    ADV_Vec y ("y", 2);
//    ADV_Negate nx (&x);
//    ADV_Sum f (y,nx);
//
//    valarray<double> result = f({ {"x", {3,4}}, {"y", {3,4}}});
//
//    cout << "result[0] = " << result[0] << "\n";
//    cout << "result[1] = " << result[1] << "\n";
//
//
//
//
//    return 0;
//}
