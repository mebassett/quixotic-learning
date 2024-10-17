#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
#include<valarray>
import autodiff;
import mnistdata;
    
using namespace std;
const unsigned int INPUT_SIZE = 784;
const unsigned int NUM_HIDDEN_NODES= 100;
const unsigned int OUTPUT_SIZE = 10;

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
        ADV_Vec* v = new ADV_Vec ("weights0-"+to_string(i), INPUT_SIZE+1);
        v->setValue(random_array(INPUT_SIZE+1, -0.05, 0.05));
        w.insert({make_pair(0,i), v});
    }
    for(int i {0}; i< OUTPUT_SIZE; i++) {
        ADV_Vec* v = new ADV_Vec ("weights1-"+to_string(i), NUM_HIDDEN_NODES);
        v->setValue(random_array(NUM_HIDDEN_NODES, -0.05, 0.05));
        w.insert({make_pair(1,i), v});
    }
}

ADV* get_predictor(Weights& w, ADV_Vec* input) {

    vector<ADV*> intermediates (NUM_HIDDEN_NODES);

    for(int i {0}; i<NUM_HIDDEN_NODES;i++){
        ADV* v = w.at(make_pair(0,i));
        ADV_InnerProduct* ip = new ADV_InnerProduct (input, v);
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

ADV* get_error(ADV* f, ADV_Vec* y) {
    ADV_Negate* nf =new ADV_Negate(f);
    ADV_Sum* sum = new ADV_Sum(y, nf);
    ADV_InnerProduct* err = new ADV_InnerProduct (sum, sum);
    ADV_Vec* scale_by_two = new ADV_Vec("const 1/2", 1);
    scale_by_two->setValue({0.5});

    ADV_VectorProduct* ret = new ADV_VectorProduct(err, scale_by_two);
    return ret;
        
}
double model_error(Training_Data& data, ADV* err ) {
    double ret = 0.0;

    for (auto& row : data) {
        ret += (*err)({ {"input", row.x}, {"target", row.t}})[0];

    }
    return ret;
}

int from_model_output(valarray<double>& out) {
  for(int i=0; i<out.size(); i++) {
    if(out[i] == out.max()) return i;
  }
  return -1;
}

int main() {
    Weights weights {};
    initialize_weights(weights);

    double learning_rate = 0.025;

    ADV_Vec input ("input",INPUT_SIZE+1);
    ADV_Vec target ("target",OUTPUT_SIZE);
    ADV* predictor = get_predictor(weights, &input);
    ADV* error = get_error(predictor, &target);

    Training_Data rows = load_data_from_file("../data/mnist_train.txt", 60000);
    Training_Data test_rows = load_data_from_file("../data/mnist_test.txt", 10000);

    int count = 1;
    int num_right = 0;
    int training_examples = 0;
    valarray<double> out;
    while(count <= 10){
        count = 99999;
        num_right = 0;
        for(auto row : test_rows) {
            out = (*predictor)( { {"input", row.x}});
            if(row.y == from_model_output(out)) num_right++;
        }
        cout << "num right: "<< num_right << " / " << test_rows.size() << ".\n";
        cout << "model error on test set: " << model_error(test_rows, error) << "\n";

        cout << "epoch " << count << "...\n";
        for(auto row : rows){
            
            (*error)({ {"input", row.x}, {"target", row.t}})[0];
            error->take_gradient({1});

            for(auto [coords, adv] : weights){ 
                valarray<double> gradient = adv->get_gradient();
                valarray<double> new_val = adv->val - (learning_rate * gradient);
                adv->setValue(new_val);
            }
            training_examples++;
            if(training_examples % 10000 == 0) cout << "training examples so far: "<< training_examples << "/"<< rows.size() << ".\n";
        }
        count++;
        training_examples = 0;

    }


    return 0;
}
