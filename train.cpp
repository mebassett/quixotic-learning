#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<random>
#include<valarray>
#include<cmath>

using namespace std;
const unsigned int INPUT_SIZE = 784;

struct Training_Datum {
  valarray<double> x;
  double t;
};

typedef vector<Training_Datum> Training_Data ;

Training_Data load_data_from_file(string filename) {
  ifstream ifs {filename};

  if(!ifs) {
      cout << "couldn't open training file";
      return {};
  }

  Training_Data rows {};

  for(string row_buffer_str; getline(ifs, row_buffer_str);) {
      istringstream row_buffer;
      valarray<double> current_row(INPUT_SIZE);

      row_buffer.str(row_buffer_str);

      int i = 0;
      
      for(double f; row_buffer>>f;) {
        if(i<INPUT_SIZE) {
          current_row[i] = f;
          i++;
        } else {

          rows.push_back({current_row, f});
        
          continue;
        }
    
      }
      current_row = {};
  }
  return rows;
}

struct Model_Weights {
  vector<valarray<double>> layer0_weights;
  vector<valarray<double>> layer1_weights;
  const unsigned int input_size;
  const unsigned int num_hidden_nodes;
  const unsigned int output_size;
};


Model_Weights initiate_weights( unsigned int input_size
                              , unsigned int num_nodes
                              , unsigned int output_size){
    using my_engine = default_random_engine;
    using my_distribution = uniform_real_distribution<>;

    my_engine eng {};
    my_distribution dist { -0.05, 0.05 };

    auto get_rand = [&](){ return dist(eng); } ; // wtf? a lambda function in c++ ?

    vector<valarray<double>> w0 {};
    vector<valarray<double>> w1 {};

    for(int j = 0; j<num_nodes;j++) {
      valarray<double> weights(input_size);
      for(int i =0; i<input_size; i++){
        weights[i] =  get_rand();
      }
      w0.push_back(weights);

    }

    for(int j=0; j<output_size;j++) {
      valarray<double> weights (num_nodes);
      for(int i =0; i<num_nodes;i++){
        weights[i] = get_rand() ;
      }
      w1.push_back(weights);
    }

    return { w0, w1, input_size, num_nodes, output_size };
}

double layer_0_activation(double x) {
  return (exp(x)  - exp(-x)) / ( exp(x) + exp(-x) );
}

double layer_0_activation_prime(double x) {
  return 1 - pow(layer_0_activation(x), 2);
}


double layer_0_output(Model_Weights& weights, unsigned int j, valarray<double>& input) {
  return layer_0_activation((weights.layer0_weights[j] * input).sum());
}

double layer_1_output(Model_Weights& weights, unsigned int j, valarray<double>& input) {
    valarray<double> output_0(weights.num_hidden_nodes);

    for(int i =0; i< weights.num_hidden_nodes; i++) {
        output_0[i] = layer_0_output(weights, i, input);
    }

    return layer_0_activation( (weights.layer1_weights[j] * output_0).sum() );
}

valarray<double> model_output(Model_Weights& weights, valarray<double>& input) {
  valarray<double> ret(weights.output_size);
  for(int j = 0; j<weights.output_size;j++) {
    ret[j] = layer_1_output(weights, j, input);
  }
  return ret;
}

int from_model_output(valarray<double>& out) {
  for(int i=0; i<out.size(); i++) {
    cout << "data: " << out[i] << " max: " << out.max() << "\n";
    if(out[i] == out.max()) return i;
  }
  return -1;
}


int main(void) {
    Training_Data rows = load_data_from_file("mnist_train.txt");
    Model_Weights weights = initiate_weights(784, 10, 10);

    for(auto row : rows) { // = rows[0];;) {

        valarray<double> model_out = model_output(weights, row.x);

        cout << from_model_output(model_out) << " : " << row.t << "\n";

    }

    return 0;
}
