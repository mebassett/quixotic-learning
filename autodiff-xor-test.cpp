#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
#include<valarray>
import autodiff;
    
using namespace std;
const unsigned int INPUT_SIZE = 2;
const unsigned int NUM_HIDDEN_NODES= 5;
const unsigned int OUTPUT_SIZE = 1;

struct Training_Datum {
  valarray<double> x;
  int y;
  valarray<double> t;
};

typedef vector<Training_Datum> Training_Data ;

Training_Data rows = {{{1,0,0}, 0, {0}}, {{1,0,1},1,{1}}, {{1,1,0},1,{1}},{{1,1,1},0,{0}}};

typedef map<pair<int,int>, ADV*> ADV_Weights;

struct FFNN_Model {
  double* weights;
  const unsigned int input_size;
  const unsigned int num_hidden_nodes;
  const unsigned int output_size;
};

valarray<double> getFromPointer(double* weights, int start, int size) {
    valarray<double> ret (size);
    for(int i {0}; i<size;i++)
        ret[i] = *(weights+start+i);
    return ret;
}

FFNN_Model initiate_weights_ ( unsigned int input_size
                              , unsigned int num_nodes
                              , unsigned int output_size
                              , ADV_Weights& w){

    int weights0_size = input_size*num_nodes;
    int weights1_size = num_nodes*output_size;
    double* weights = new double[weights0_size+weights1_size];

    random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(0.095, 0.095000001);


    auto get_rand = [&](){ return dis(rd); } ;
    for(int i {0}; i < weights0_size+weights1_size;i++){
        *(weights+i) = get_rand();
    }

    FFNN_Model scratch_model  {weights, input_size, num_nodes, output_size };

    for(int i {0}; i< NUM_HIDDEN_NODES; i++) {
        ADV_Vec* v = new ADV_Vec ("weights0-"+to_string(i), input_size);
        v->setValue(getFromPointer(weights, input_size*i, input_size));
        w.insert({make_pair(0,i), v});
    }
    for(int i {0}; i< OUTPUT_SIZE; i++) {
        ADV_Vec* v = new ADV_Vec ("weights1-"+i, num_nodes);
        v->setValue(getFromPointer(weights, weights0_size+output_size*i, num_nodes));
        w.insert({make_pair(1,i), v});
    }

    return scratch_model; 
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

double layer_0_activation(double x) {
  //return  tanh(x);
  //return 1 / (1+ exp(-x));
  if(x>0) return x;
  return 0.001*x;
}

double layer_0_activation_prime(double x) {
  //return 1 - pow(layer_0_activation(x), 2);
  //return layer_0_activation(x) * (1-layer_0_activation(x));
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

struct LayerOutput {
  double output;
  double dot_product;
  double activation_prime;

  double operator-(double subtractand) {
    return this->output - subtractand;
  }

  bool operator==(LayerOutput d) {
    return d.output == this->output;
  }

  bool operator<(LayerOutput d) {
    return this->output < d.output;
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
    double dotproduct {0};
    for (int i {0}; i< model.input_size; i++)
        dotproduct += *((*rel_weights) + i) * input[i]; 
    delete rel_weights;

    return {layer_0_activation(dotproduct), dotproduct, layer_0_activation_prime(dotproduct)};
}

void all_layer_0_outputs( FFNN_Model& model
                        , valarray<double>& input
                        , valarray<LayerOutput>& output_0) {
    for(int i {0}; i < model.num_hidden_nodes; i++ )
        output_0[i] = layer_0_output(model, i, input);
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

void model_output( FFNN_Model& model 
                 , valarray<LayerOutput>& output_0
                 , valarray<LayerOutput>& output_1) {

    for(int j { 0 }; j < model.output_size; j++)
        output_1[j] = layer_1_output(model, output_0, j);

     
}

double from_model_output(valarray<LayerOutput>& out) {
  return out[0].output;
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
        model_output(model, output_0, model_out);

        sigma = row.t - model_out;
        sigma *= sigma;
        err += sigma.sum();

    }
    return 0.5*err;
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

ADV* get_predictor(ADV_Weights& w, ADV_Vec* input) {

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
    return err;
        
}

int main() {
    ADV_Weights weights {};

    int num_right_scratch = 0;
    int num_right_adv = 0;
    
    FFNN_Model model = initiate_weights_(INPUT_SIZE+1, NUM_HIDDEN_NODES, OUTPUT_SIZE, weights);

    for(auto row : rows) {

        valarray<LayerOutput> model_out (model.output_size), output_0 (model.num_hidden_nodes);
        all_layer_0_outputs(model, row.x, output_0);
        model_output(model, output_0, model_out);

        if(row.y == lround(from_model_output(model_out))) num_right_scratch++;

    }
    cout << "num right on scratch model: " << num_right_scratch << "/" << rows.size() << "\n";
    cout << "scratch model mean squared error: " << model_error(rows, model) << "\n";

    double scratch_error = 99;
    int epochs = 1;
    int MAX_EPOCH = 10000;

    while(epochs <= MAX_EPOCH){
        for(auto row : rows) {
            train_weights(model, row, -0.05);
        }
        scratch_error = model_error(rows, model);
        cout << "epoch " << epochs << " finished.\n";
        cout << "scratch model mean squared error:: " << scratch_error << "\n";

        epochs++;
    }
    return 0;
}
