#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
import autodiff;
import mnistdata;
    
using namespace std;
const unsigned int INPUT_SIZE = 784;
const unsigned int NUM_HIDDEN_NODES= 300;
const unsigned int OUTPUT_SIZE = 10;

typedef map<pair<int,int>, ADV*> Weights;

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
        ADV_Vec* v = new ADV_Vec ("weights0-"+i, INPUT_SIZE+1);
        v->setValue(random_array(INPUT_SIZE+1, -0.05, 0.05));
        w.insert({make_pair(0,i), v});
    }
    for(int i {0}; i< OUTPUT_SIZE; i++) {
        ADV_Vec* v = new ADV_Vec ("weights1-"+i, NUM_HIDDEN_NODES);
        v->setValue(random_array(NUM_HIDDEN_NODES, -0.05, 0.05));
        w.insert({make_pair(1,i), v});
    }
}

ADV* get_predict(Weights& w, ADV_Vec& input) {

    vector<ADV*> intermediates (NUM_HIDDEN_NODES);

    for(int i {0}; i<NUM_HIDDEN_NODES;i++){
        ADV_InnerProduct ip (input, *w.at(make_pair(0,i)));
        ADV_LeakyReLU relu (&ip);
        intermediates[i] = &relu;
    }

    ADV_Concat hidden_nodes ( intermediates );

    vector<ADV*> finals (OUTPUT_SIZE);

    for(int i {0}; i<OUTPUT_SIZE;i++) {
        ADV_InnerProduct ip (hidden_nodes, *w.at(make_pair(1,i)));
        finals[i] = &ip;
    }

    ADV_Concat* ret = new ADV_Concat (finals);
    return ret;
  
}

ADV* get_error(ADV* f, Weights& w, ADV_Vec& y) {
    ADV_Negate noutput (&y);
    ADV_Sum sum (*f, y);
    ADV_InnerProduct* err = new ADV_InnerProduct (sum, sum);
    return err;
        
}


int main() {
    Weights weights {};
    initialize_weights(weights);

    for( auto [p,_v]: weights) {
        ADV* v = weights.at(p);
        valarray<double> t = (*v)();
        cout << p.first << ":" << p.second << "=" << t[0] <<"\n";
    }
    //ADV* v = weights.at(make_pair(0,3));
    //
    //
    for (auto [p,v] : weights) {
        delete v;
    }

    return 0;
}
