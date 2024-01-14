#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<random>
#include<valarray>
#include<cmath>

using namespace std;
const unsigned int INPUT_SIZE = 2;//784;
const unsigned int OUTPUT_SIZE = 1;//10;


valarray<double> to_model_output(int in) {
    valarray<double> ret(OUTPUT_SIZE);
    for(int i = 0; i<OUTPUT_SIZE; i++)
      if(i==in)
        ret[i] = 1;
      else 
        ret[i] = 0;
    return ret;

}

struct Training_Datum {
  valarray<double> x;
  int y;
  valarray<double> t;
};

typedef vector<Training_Datum> Training_Data ;

void print_out(Training_Datum& row) {
      cout << "OUT " << row.y << ": ";
      for(auto i : row.t) {
          cout << i << ", ";
      } 

      cout << "\n";
}

void print_outs(Training_Data& rows) {
    for (auto row : rows) {
        print_out(row);
    }
}

Training_Data load_data_from_file(string filename) {
  ifstream ifs {filename};

  if(!ifs) {
      cout << "couldn't open training file";
      return {};
  }

  Training_Data rows {};

  for(string row_buffer_str; getline(ifs, row_buffer_str);) {
      istringstream row_buffer;
      valarray<double> current_row(INPUT_SIZE+1);

      row_buffer.str(row_buffer_str);

      int i = 0;
      double max = 0;
      
      for(double f; row_buffer>>f;) {
        if(i<INPUT_SIZE) {
          current_row[i+1] = f;
          if (f > max) max = f;
          i++;
        } else {
          if(max != 0)
              current_row = (1.0/max) * current_row; 
          current_row[0] = 1.0;

          rows.push_back({current_row, lround(f), { f } } );//to_model_output(lround(f))});
        
          continue;
        }
    
      }
      current_row = {};
      if(rows.size() >= 200) return rows;
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

    random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(-1.05, 1.05);


    auto get_rand = [&](){ return dis(rd); } ; // wtf? a lambda function in c++ ?

    vector<valarray<double>> w0;
    vector<valarray<double>> w1;

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
  //return  tanh(x);
  return 1 / (1+ exp(-x));
  if(x>0) return x;
  return 0.001*x;
}

double layer_0_activation_prime(double x) {
  //return 1 - pow(layer_0_activation(x), 2);
  return layer_0_activation(x) * (1-layer_0_activation(x));
  if(x>0) return 1;
  return 0.01;
}

double layer_1_activation(double x) {
  //return 1 / (1+ exp(-x));
  return x;
}

double layer_1_activation_prime(double x) {
  //return layer_1_activation(x) * (1-layer_1_activation(x));
  return 1;
}


double layer_0_output(Model_Weights& weights, unsigned int j, valarray<double>& input) {
  return layer_0_activation((weights.layer0_weights[j] * input).sum());
}

valarray<double> all_layer_0_outputs(Model_Weights& weights, valarray<double>& input) {
    valarray<double> output_0(weights.num_hidden_nodes);
    for(int i =0; i< weights.num_hidden_nodes; i++) {
        output_0[i] = layer_0_output(weights, i, input);
    }
    return output_0;

}

double layer_1_output(Model_Weights& weights, unsigned int j, valarray<double>& input) {
    valarray<double> output_0 = all_layer_0_outputs(weights, input);


    return layer_1_activation( (weights.layer1_weights[j] * output_0).sum() );
}

valarray<double> model_output(Model_Weights& weights, valarray<double>& input) {
  valarray<double> ret(weights.output_size);
  for(int j = 0; j<weights.output_size;j++) {
    ret[j] = layer_1_output(weights, j, input);
  }
  return ret;
}

double from_model_output(valarray<double>& out) {
  return out[0];
  for(int i=0; i<out.size(); i++) {
    if(out[i] == out.max()) return i;
  }
  return -1;
}

double model_error(Training_Data& data, Model_Weights& weights) {
    double err = 0.0;

    for (auto row : data) {
        valarray<double> sigma = row.t - model_output(weights, row.x);
        sigma *= sigma;
        err += sigma.sum();
    }

    return 0.5*err;

}

double del_err_by_del_weight_1_I_J (Model_Weights& weights, int i, int j, Training_Datum& row, double activation_result) {
    double sigma_j = layer_1_output(weights, j, row.x);
    return - ( row.t[j] - sigma_j) 
           * activation_result // layer_1_activation_prime( ( all_layer_0_outputs(weights, row.x) * weights.layer1_weights[j]  ).sum()  ) 
           * layer_0_output(weights, i, row.x);
}

double del_err_by_weight_0_K_J (Model_Weights& weights, int K, int J, Training_Datum& row,
        valarray<double> activation_vector ) {
    double iter = 0.0;


    valarray<double> relevant_weights ( weights.output_size);
    for(int j = 0 ; j< weights.output_size;j++) {
        relevant_weights[j] = weights.layer1_weights[j][J];

    }
    return - layer_0_activation_prime(( weights.layer0_weights[J] * row.x).sum())
           * (activation_vector * relevant_weights).sum()
           * layer_0_output(weights, J, row.x)
           * row.x[K];

}

bool train_weights(Model_Weights& weights, Training_Datum& row, double learning_rate) {
    vector<valarray<double>> grad_err_by_layer_1 (weights.layer1_weights.size());
    vector<valarray<double>> grad_err_by_layer_0 (weights.layer0_weights.size());


    for(int j=0;j<weights.layer1_weights.size(); j++) {
        grad_err_by_layer_1[j] = valarray<double>(weights.num_hidden_nodes);
        double activation_result = layer_1_activation_prime( ( all_layer_0_outputs(weights, row.x) * weights.layer1_weights[j]  ).sum()  ) ;
        for(int i =0; i<weights.num_hidden_nodes;i++) {
            grad_err_by_layer_1[j][i] = del_err_by_del_weight_1_I_J(weights, i, j, row, activation_result);

        }


        

    }


    valarray<double> activation_vector = row.t - model_output(weights, row.x);

    valarray<double> activation_prime_vector (row.t.size());

    for (int i = 0; i< row.t.size(); i++) {
      activation_prime_vector[i] =  
        layer_1_activation_prime(( weights.layer1_weights[i] * all_layer_0_outputs(weights, row.x)   ).sum());
    }

    activation_vector *= activation_prime_vector;

    for (int J=0; J<weights.layer0_weights.size(); J++) {
        grad_err_by_layer_0[J] = valarray<double>(weights.layer1_weights.size());
        for(int K=0; K<weights.input_size;K++) {
          double t =   del_err_by_weight_0_K_J(weights, K, J, row, activation_vector);
          grad_err_by_layer_0[J][K] = t;
        }

    }

    for(int i = 0;i<weights.layer0_weights.size();i++)
      weights.layer0_weights[i] += learning_rate * grad_err_by_layer_0[i]; 

    for(int i = 0;i<weights.layer1_weights.size();i++)
      weights.layer1_weights[i] += learning_rate * grad_err_by_layer_1[i]; 


    return true;
}

void print_weights(Model_Weights& weights) {
  cout << "printing weights...\nlayer0 weights:\n";

  for (int i=0; i < weights.layer0_weights.size(); i++) {
    cout << "W^{0, ? }_" << i <<" = [";
    for (auto x : weights.layer0_weights[i]) 
      cout << x << ", ";
    cout << "\n";
  }
  cout << "layer 1 weights:\n";
  for (int i=0; i < weights.layer1_weights.size(); i++) {
    cout << "W^{1, ? }_" << i <<" = [";
    for (auto x : weights.layer1_weights[i]) 
      cout << x << ", ";
    cout << "\n";
  }
}



int main(void) {
    Training_Data rows = load_data_from_file("xor_train.tsv");
    Model_Weights weights = initiate_weights(INPUT_SIZE+1, 2, OUTPUT_SIZE);


    cout << "model mean squared error on training data: " << model_error(rows, weights) << "\n";
    int count = 0;
    double error = 9999;
    int rounds = 1;
    for(auto row : rows) { // = rows[0];;) {

        valarray<double> model_out = model_output(weights, row.x);

        cout << from_model_output(model_out) << " : " << row.y << "\n";
        if(row.y == round(from_model_output(model_out))) count++;

    }
    cout << "num right: " << count << "/" << rows.size() << "\n";
    count = 0;
    while(error > 0.05) {
        print_weights(weights);

        for(auto row : rows) {
            train_weights(weights, row, -0.5);
        }
        error = model_error(rows, weights);
        cout << "round " << rounds << " finished.\n";
        cout << "model mean squared error on training data: " << error << "\n";
        for(auto row : rows) { // = rows[0];;) {

            valarray<double> model_out = model_output(weights, row.x);

            cout << from_model_output(model_out) << " : " << row.y << "\n";
            if(row.y == lround(from_model_output(model_out))) count++;

        }
    cout << "num right: " << count << "/" << rows.size() << "\n";
    count = 0;
    rounds++;
    }

    return 0;
}
