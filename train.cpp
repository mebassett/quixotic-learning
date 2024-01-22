#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<random>
#include<valarray>
#include<cmath>

using namespace std;
const unsigned int INPUT_SIZE = 784;
const unsigned int OUTPUT_SIZE = 10;


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

          rows.push_back({current_row, lround(f), to_model_output(lround(f))});
        
          continue;
        }
    
      }
      current_row = {};
      if(rows.size() >= 10) return rows;
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

struct FFNN_Model {
  double* weights;
  const unsigned int input_size;
  const unsigned int num_hidden_nodes;
  const unsigned int output_size;
};

FFNN_Model initiate_weights_ ( unsigned int input_size
                              , unsigned int num_nodes
                              , unsigned int output_size){

    int weights0_size = input_size*num_nodes;
    int weights1_size = num_nodes*output_size;
    double* weights = new double[weights0_size+weights1_size];

    random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(-10.05, 10.05);


    auto get_rand = [&](){ return dis(rd); } ;
    for(int i {0}; i < weights0_size+weights1_size;i++){
        *(weights+i) = i*1.0;//get_rand();
    }

    return {weights, input_size, num_nodes, output_size };
}

double** get_weights0_to_j( FFNN_Model& model
                          , unsigned int j) {
  double** ret = new double*[model.input_size];

  for(int i {0}; i < model.input_size; i++) {
    *(ret+i) = (model.weights + j*model.input_size + i);
  }

  return ret;
}

double** get_weights1_to_j( FFNN_Model& model
                          , unsigned int j ) {
    double** ret = new double*[model.num_hidden_nodes];

    for(int i {0}; i < model.num_hidden_nodes; i++) {
      *(ret+i) = 
        model.weights + (model.input_size * model.num_hidden_nodes 
                        + j*model.num_hidden_nodes + i);
    }
    return ret;
}



Model_Weights initiate_weights( unsigned int input_size
                              , unsigned int num_nodes
                              , unsigned int output_size){

    random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(-10.05, 10.05);


    auto get_rand = [&](){ return dis(rd); } ; // wtf? a lambda function in c++ ?

    vector<valarray<double>> w0;
    vector<valarray<double>> w1;

    for(int j = 0; j<num_nodes;j++) {
      valarray<double> weights(input_size);
      for(int i =0; i<input_size; i++){
        weights[i] =  (j*input_size + i) *1.0 ;//get_rand();
      }
      w0.push_back(weights);

    }

    for(int j=0; j<output_size;j++) {
      valarray<double> weights (num_nodes);
      for(int i =0; i<num_nodes;i++){
        weights[i] = (j*num_nodes + i + input_size*num_nodes) * 1.0;//get_rand() ;
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

struct LayerOutput {
  double output;
  double dot_product;
  double activation_prime;

  double operator-(double subtractand) {
    return this->output - subtractand;
  }
};

double operator-(double d, LayerOutput& l) {
    return d - l.output;
}

valarray<double> operator-(valarray<double>& sd, valarray<LayerOutput>& sl) {
    valarray<double> ret (sd.size());
    for(int i {0};i<sd.size();i++){
        ret[i] = sd[i] - sl[i];
    }
    return ret;
}

LayerOutput layer_0_output(FFNN_Model& model, unsigned int j, valarray<double>& input) {
    double** rel_weights = get_weights0_to_j(model, j);
    double ret {1};
    for (int i {0}; i< model.input_size; i++)
        ret += *((*rel_weights) + 1) * input[i]; 
    delete rel_weights;

    return {layer_0_activation(ret), ret, layer_0_activation_prime(ret)};
}

valarray<double> all_layer_0_outputs(Model_Weights& weights, valarray<double>& input) {
    valarray<double> output_0(weights.num_hidden_nodes);
    for(int i =0; i< weights.num_hidden_nodes; i++) {
        output_0[i] = layer_0_output(weights, i, input);
    }
    return output_0;

}


void all_layer_0_outputs( FFNN_Model& model
                        , valarray<double>& input
                        , valarray<LayerOutput>& output_0) {
    for(int i {0}; i < model.num_hidden_nodes; i++ )
        output_0[i] = layer_0_output(model, i, input);
}

double layer_1_output(Model_Weights& weights, unsigned int j, valarray<double>& input) {
    valarray<double> output_0 = all_layer_0_outputs(weights, input);
    return layer_1_activation( (weights.layer1_weights[j] * output_0).sum() );
}

LayerOutput layer_1_output( FFNN_Model& model
                   , valarray<LayerOutput>& output_0_for_input
                   , unsigned int j) {
    double** weights { get_weights1_to_j(model, j) };
    double ret {0};

    for(int i {0}; i<model.num_hidden_nodes;i++) 
        ret += *(*weights+i) * output_0_for_input[i].output;

    delete weights;

    return { layer_1_activation ( ret ), ret, layer_1_activation_prime(ret) };

}


valarray<double> model_output(Model_Weights& weights, valarray<double>& input) {
  valarray<double> ret(weights.output_size);
  for(int j = 0; j<weights.output_size;j++) {
    ret[j] = layer_1_output(weights, j, input);
  }
  return ret;
}

void model_output( FFNN_Model& model 
                 , valarray<LayerOutput>& output_0
                 , valarray<LayerOutput>& output_1) {

    for(int j { 0 }; j < model.output_size; j++)
        output_1[j] = layer_1_output(model, output_0, j);

     
}

double from_model_output(valarray<double>& out) {
  for(int i=0; i<out.size(); i++) {
    if(out[i] == out.max()) return i;
  }
  return -1;
}

double model_error(Training_Data& data, FFNN_Model& model) {
    double err = 0.0;
    valarray<LayerOutput> model_out (model.output_size); 
    valarray<double> sigma(model.output_size);
    valarray<LayerOutput> output_0 (model.num_hidden_nodes);

    for (auto& row : data) {
        all_layer_0_outputs(model, row.x, output_0);
        //model_output(model, output_0, model_out);

        //sigma = row.t - model_out;
        //sigma *= sigma;
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

void train_weights( FFNN_Model& model
                  , Training_Datum& row
                  , double learning_rate ) {
    int weights0_size = model.input_size * model.num_hidden_nodes;
    int weights1_size = model.num_hidden_nodes * model.output_size;
    double grad_weights[weights0_size + weights1_size] {};

    valarray<LayerOutput> output_0 (model.num_hidden_nodes);
    all_layer_0_outputs(model, row.x, output_0);

    valarray<LayerOutput> output_1 (model.output_size);
    model_output(model, output_0, output_1);

    valarray<double> output_1_out ( model.output_size);
    valarray<double> output_1_weights ( model.output_size);
    valarray<double> output_1_prime ( model.output_size);
    for(int i {0}; i<model.output_size;i++){
        output_1_out[i] = output_1[i].output;
        output_1_weights[i] = output_1[i].dot_product;
        output_1_prime[i] = output_1[i].activation_prime;
    }



    valarray<double> errors = row.t - output_1_out;

    valarray<double> weight0_activation_vector =
        errors * output_1_prime;


    for(int j {0}; j < model.num_hidden_nodes; j++){
        
        valarray<double> relevant_weights ( model.output_size);
        for(int l {0}; l<model.output_size;l++){
            unsigned int index = weights0_size + l*model.num_hidden_nodes + j;
            relevant_weights[l] = *(model.weights+index);
        }

        double activation_term =
          (relevant_weights * weight0_activation_vector).sum();

        for(int k {0}; k < model.input_size; k++){
            unsigned int index { j*model.input_size+k };

            grad_weights[index] = - learning_rate
                                * activation_term
                                * output_0[j].output
                                * output_0[j].activation_prime
                                * row.x[k];
        }
    }

    for(int j {0}; j<model.output_size; j++){
        for(int i {0}; i<model.num_hidden_nodes;i++){
            int index = weights0_size + j*model.num_hidden_nodes + i;

            grad_weights[index] = - learning_rate
                                * errors[j]
                                * output_1_prime[j]
                                * output_0[i].output;
        }
    }

    for(int i {0}; i < weights0_size + weights1_size; i++) {
        *(model.weights+i) = *(model.weights+i) + grad_weights[i];
    }

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
        grad_err_by_layer_0[J] = valarray<double>(weights.input_size);
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
    Training_Data rows = load_data_from_file("mnist_train.txt");
    FFNN_Model model = initiate_weights_(INPUT_SIZE+1, 28, OUTPUT_SIZE);

    Model_Weights weights = initiate_weights(INPUT_SIZE+1, 28, OUTPUT_SIZE);

    for(auto row : rows) {
        train_weights(model, row, -0.05);
        train_weights(weights, row, -0.05);
    }

    for(int i {0};i < 10; i++){
      cout << "model weight " << i << " is " << *(model.weights+i) << "\n";
    }

    double** test = get_weights1_to_j(model, 5);
    for(int i {0}; i < 28; i++) {
        cout << "new model weight to j from " << i << ": " << *((*test)+i) 
             << " old model weight to j from " << i << ": "
             << weights.layer1_weights[5][i]
             << "\n";
    }

    cout << "output^0_0 = " << layer_0_output(model, 1, rows[0].x).output << "\n";
    cout << "output^0_0 = " << layer_0_output(weights, 1, rows[0].x) << "\n";

    valarray<double> output_0_org = all_layer_0_outputs(weights, rows[0].x);
    valarray<LayerOutput> output_0_new (28);
    all_layer_0_outputs(model, rows[0].x, output_0_new);

    cout << "\n\noutput 0:\n";
    for(int i {0}; i<28; i++){
        cout << i << " org says: " 
             << output_0_org[i]
             << " new says: "
             << output_0_new[i].output
             << "\n";
    }

    for(int i {0}; i< 10; i++) {
        valarray<double> output_org { model_output(weights, rows[i].x) };

        valarray<LayerOutput> output_0 (28);
        all_layer_0_outputs(model, rows[i].x, output_0);
        valarray<LayerOutput> output_new (model.output_size);
        model_output(model, output_0, output_new);

        cout << "\ncomparing the " << i << "th output..\n";
        for(int j {0};j < model.output_size;j++)
            cout << j << " org says: " << output_org[j] << " new says: " 
                 << output_new[j].output << "\n";
    }


//    return 0;
//}


    cout << "model mean squared error on training data: " << model_error(rows, model) << "\n";
    //int count = 0;
    //double error = 9999;
    //int rounds = 1;
    //for(auto row : rows) { // = rows[0];;) {

    //    valarray<double> model_out = model_output(weights, row.x);

    //    cout << from_model_output(model_out) << " : " << row.y << "\n";
    //    if(row.y == round(from_model_output(model_out))) count++;

    //}
    //cout << "num right: " << count << "/" << rows.size() << "\n";
    //count = 0;
    //while(error > 0.05) {
    //    //print_weights(weights);

    //    for(auto row : rows) {
    //        train_weights(weights, row, -0.05);
    //    }
    //    error = model_error(rows, weights);
    //    cout << "round " << rounds << " finished.\n";
    //    cout << "model mean squared error on training data: " << error << "\n";
    //    for(auto row : rows) { // = rows[0];;) {

    //        valarray<double> model_out = model_output(weights, row.x);

    //        cout << from_model_output(model_out) << " : " << row.y << "\n";
    //        if(row.y == lround(from_model_output(model_out))) count++;

    //    }
    //cout << "num right: " << count << "/" << rows.size() << "\n";
    //count = 0;
    //rounds++;
    //}

    return 0;
}
