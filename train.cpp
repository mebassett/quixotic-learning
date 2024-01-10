#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<random>
#include<valarray>

using namespace std;

typedef vector<vector<double>> Training_Data ;

Training_Data load_data_from_file(string filename) {
  ifstream ifs {filename};

  if(!ifs) {
      cout << "couldn't open training file";
      return {};
  }

  Training_Data rows;

  for(string row_buffer_str; getline(ifs, row_buffer_str);) {
      istringstream row_buffer;
      vector<double> current_row {};

      row_buffer.str(row_buffer_str);
      
      for(double f; row_buffer>>f;)
          current_row.push_back(f);

      rows.push_back(current_row);
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

int main(void) {


    //Training_Data rows = load_data_from_file("mnist_train.txt");
    Model_Weights weights = initiate_weights(784, 10, 10);

    for(auto j: weights.layer0_weights) {
      for(auto i : j) {
        cout << i << "\n";
      }
    }



    return 0;
}
