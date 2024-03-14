#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
#include<valarray>
import autodiff;
import mnistdata;
    
using namespace std;
const unsigned int INPUT_SIZE = 2;//784;
const unsigned int NUM_HIDDEN_NODES= 4;//300;
const unsigned int OUTPUT_SIZE = 1;//10;

typedef map<pair<int,int>, ADV*> Weights;

ostream& operator<<(ostream& os, valarray<double>& vec){
    os << "{ ";
    for(int i {0};i<vec.size();i++) {
        os << vec[i] << ", ";
    }
    os << " }\n";
    return os;
}

valarray<double> random_array(unsigned int size, double lower_bound, double upper_bound) {
    valarray<double> ret (size);
    random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    auto get_rand = [&](){ return dis(rd); } ;

    for(int i {0}; i<size; i++)
        ret[i] = get_rand();
    return ret;
}

void initialize_weights(Weights& w) {

    for(int i {0}; i< NUM_HIDDEN_NODES; i++) {
        ADV_Vec* v = new ADV_Vec ("weights0-"+to_string(i), INPUT_SIZE);
        v->setValue(random_array(INPUT_SIZE, -0.05, 0.05));
        w.insert({make_pair(0,i), v});
    }
    for(int i {0}; i< OUTPUT_SIZE; i++) {
        ADV_Vec* v = new ADV_Vec ("weights1-"+i, NUM_HIDDEN_NODES);
        v->setValue(random_array(NUM_HIDDEN_NODES, -0.05, 0.05));
        w.insert({make_pair(1,i), v});
    }
}

ADV* get_predictor(Weights& w, ADV_Vec& input) {

    vector<ADV*> intermediates (NUM_HIDDEN_NODES);

    for(int i {0}; i<NUM_HIDDEN_NODES;i++){
        ADV* v = w.at(make_pair(0,i));
        ADV_InnerProduct* ip = new ADV_InnerProduct (&input, v);
        ADV_LeakyReLU* relu  = new ADV_LeakyReLU (ip);
        intermediates[i] = relu;
    }

    ADV_Concat* hidden_nodes = new ADV_Concat( intermediates );

    vector<ADV*> finals (OUTPUT_SIZE);

    for(int i {0}; i<OUTPUT_SIZE;i++) {
        ADV_InnerProduct* ip = new ADV_InnerProduct(hidden_nodes, w.at(make_pair(1,i)));
        finals[i] = ip;
    }

    ADV_Concat* ret = new ADV_Concat (finals);
    return ret;
  
}

ADV* get_error(ADV* f, Weights& w, ADV_Vec* y) {
    ADV_Negate* noutput =new ADV_Negate(f);
    ADV_Sum* sum = new ADV_Sum(y, f);
    ADV_InnerProduct* err = new ADV_InnerProduct (sum, sum);
    return err;
        
}


int main() {
    Weights weights {};
    initialize_weights(weights);

    vector<valarray<double>> xs = {{1,1},{0,0},{0,1},{1,0}};
    vector<valarray<double>> ys = {{0}, {0}, {1}, {1}};
    double learning_rate = 0.1;

    for( auto [p,_v]: weights) {
        ADV* v = weights.at(p);
        valarray<double> t = (*v)();
        cout << p.first << ":" << p.second << "=" << t[0] <<"\n";
    }


    ADV_Vec input ("input",INPUT_SIZE);
    ADV_Vec target ("target",OUTPUT_SIZE);
    ADV* predictor = get_predictor(weights, input);
    ADV* error = get_error(predictor, weights, &target);

    double err =0;
    while(err == err){
        cout << "predicting...\n";
        for(int i {0}; i<4;i++){
            double y_pred = (*predictor)({ {"input", xs[i]}})[0];
            double y_real = ys[i][0];
            cout << "predicting for row " << i << " real value: " << y_real
                 << "predicted value: " << y_pred << "\n";
        }
        cout << "training...\n";
        for(int i {0}; i<4;i++){
            err = (*error)({ {"input", xs[i]}, {"target", ys[i]}})[0];
            cout << "error for " << i << ": " << err << "\n";
            error->take_gradient({1});
            for(auto [coords, adv] : weights){ 
                valarray<double> gradient = adv->get_gradient();
                valarray<double> new_val = adv->val - (learning_rate * gradient);
                cout << "gradient for (" << coords.first << "," << coords.second  << ") "
                     << gradient
                     << " and old val for weights "
                     << adv->val
                     << " and new val for weights "
                     << new_val
                     << "\n";
                adv->setValue(new_val);
            }
        }
        cout << "done!\n";

    }


    return 0;
}
