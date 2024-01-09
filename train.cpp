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
  valarray<double> layer0_weights;
  valarray<double> layer1_weights;
  const unsigned int input_size;
  const unsigned int num_hidden_nodes;
};


Model_Weights initiate_layer0_weights(unsigned int input_size, unsigned int num_nodes){
    using my_engine = default_random_engine;
    using my_distribution = uniform_real_distribution<>;

    my_engine eng {};
    my_distribution dist { -0.05, 0.05 };

    auto get_rand = [&](){ return dist(eng); } ; // wtf? a lambda function in c++ ?



    return { get_rand() };
}

int main(void) {


    Training_Data rows = load_data_from_file("mnist_train.txt");


    cout << rows.size() << "\n\nall done!\n\n";
    cout << rows[0].size() << "\n\nall done!\n\n";


     for(const vector<double> row : rows) { 
       cout << row[784] << "\n";
    //   for(const float cell : row) 
    //     cout << cell << "\t";

    //   cout << "\n";
     }

    return 0;
}
