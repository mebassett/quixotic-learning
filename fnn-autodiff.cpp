#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
import autodiff;
import mnistdata;
    
using namespace std;
const unsigned int INPUT_SIZE = 2;//784;
const unsigned int NUM_HIDDEN_NODES= 10;//300;
const unsigned int OUTPUT_SIZE = 1;//10;

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
        ADV_Vec* v = new ADV_Vec ("weights0-"+to_string(i), INPUT_SIZE+1);
        v->setValue(random_array(INPUT_SIZE+1, -0.05, 0.05));
        w.insert({make_pair(0,i), v});
    }
    for(int i {0}; i< OUTPUT_SIZE; i++) {
        ADV_Vec* v = new ADV_Vec ("weights1-"+i, NUM_HIDDEN_NODES);
        v->setValue(random_array(NUM_HIDDEN_NODES, -0.05, 0.05));
        w.insert({make_pair(1,i), v});
    }
}

ADV* get_predictor(Weights& w, ADV_Vec& input) {

    cout << "starting predictor...\n";
    vector<ADV*> intermediates (NUM_HIDDEN_NODES);

    for(int i {0}; i<NUM_HIDDEN_NODES;i++){
        ADV* v = w.at(make_pair(0,i));
        ADV_InnerProduct* ip = new ADV_InnerProduct (&input, v);
        ADV_LeakyReLU* relu  = new ADV_LeakyReLU (ip);
        intermediates[i] = relu;
    }
    cout << "done with intermediates ...\n";

    ADV_Concat* hidden_nodes = new ADV_Concat( intermediates );
    cout << "done with intermediate concat ...\n";

    vector<ADV*> finals (OUTPUT_SIZE);

    for(int i {0}; i<OUTPUT_SIZE;i++) {
        ADV_InnerProduct* ip = new ADV_InnerProduct(hidden_nodes, w.at(make_pair(1,i)));
        finals[i] = ip;
    }
    cout << "done with finals ...\n";

    ADV_Concat* ret = new ADV_Concat (finals);
    cout << "done with final concat ...\n";
    return ret;
  
}

ADV* get_error(ADV* f, Weights& w, ADV_Vec* y) {
    ADV_Negate* noutput =new ADV_Negate(y);
    ADV_Sum* sum = new ADV_Sum(f, y);
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


    ADV_Vec input ("input",INPUT_SIZE+1);
    ADV_Vec target ("target",OUTPUT_SIZE);
    cout <<"calling predictor...\n";
    ADV* predictor = get_predictor(weights, input);
    cout << "calling error\n";
    ADV* error = get_error(predictor, weights, &target);
    cout << "using predictor...\n";

    double y = (*predictor)({ {"input", {1,1,1}}})[0];

    cout << "y value is : " << y << "\n";


    //ADV* v = weights.at(make_pair(0,3));
    //
    //
    for (auto [p,v] : weights) {
        delete v;
    }

    return 0;
}
