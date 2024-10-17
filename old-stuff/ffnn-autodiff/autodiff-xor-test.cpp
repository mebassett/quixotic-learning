#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
#include<valarray>
import autodiff;
    
using namespace std;
const unsigned int INPUT_SIZE = 2;
const unsigned int NUM_HIDDEN_NODES= 1;
const unsigned int OUTPUT_SIZE = 1;

struct Training_Datum {
  valarray<double> x;
  int y;
  valarray<double> t;
};

typedef vector<Training_Datum> Training_Data ;

//Training_Data rows = {{{1,0,0}, 0, {0}}, {{1,0,1},1,{1}}, {{1,1,0},1,{1}},{{1,1,1},0,{0}}};
Training_Data rows = {{{1,1}, 0, {0}}, {{0,1},1,{1}}, {{1,0},1,{1}},{{0,0},0,{0}}};

typedef map<pair<int,int>, ADV*> ADV_Weights;

struct FFNN_Model {
  double* weights;
  const unsigned int input_size;
  const unsigned int num_hidden_nodes;
  const unsigned int output_size;
};

ostream& operator<<(ostream& os, valarray<double>& vec){
    os << "{ ";
    for(int i {0};i<vec.size();i++) {
        os << vec[i] << ", ";
    }
    os << " }";
    return os;
}

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
    uniform_real_distribution<> dis(0.45, 0.55000001);


    auto get_rand = [&](){ return (double)(round(dis(rd))*2); } ;
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
        ADV_Vec* v = new ADV_Vec ("weights1-"+to_string(i), num_nodes);
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
  return 0.01*x;
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

double model_error(Training_Data& data, ADV* err ) {
    double ret = 0.0;

    for (auto& row : data) {
        ret += (*err)({ {"input", row.x}, {"target", row.t}})[0];

    }
    return ret;
}

valarray<double> train_weights( FFNN_Model& model
                  , Training_Datum& row
                  , double learning_rate ) {
    int weights0_size = model.input_size * model.num_hidden_nodes;
    int weights1_size = model.num_hidden_nodes * model.output_size;
    valarray<double> grad_weights (weights0_size + weights1_size);

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

            grad_weights[index] = - activation_term
                                * output_0[j].output
                                * output_0[j].activation_prime
                                * row.x[k];
        }
    }

    for(int j {0}; j<model.output_size; j++){
        for(int i {0}; i<model.num_hidden_nodes;i++){
            int index = weights0_size + j*model.num_hidden_nodes + i;

            grad_weights[index] = - errors[j]
                                * output_1_prime[j]
                                * output_0[i].output;
        }
    }

    for(int i {0}; i < weights0_size + weights1_size; i++) {
        *(model.weights+i) = *(model.weights+i) - learning_rate * grad_weights[i];
    }
    return grad_weights;

}

ADV* get_predictor(ADV_Weights& w, ADV_Vec* input) {

    vector<ADV*> intermediates (NUM_HIDDEN_NODES);

    for(int i {0}; i<NUM_HIDDEN_NODES;i++){
        ADV_InnerProduct* ip = new ADV_InnerProduct (input, w.at(make_pair(0,i)));
        intermediates[i] = new ADV_LeakyReLU(ip);
    }

    ADV_Concat* hidden_nodes = new ADV_Concat( intermediates );

    vector<ADV*> finals (OUTPUT_SIZE);

    for(int i {0}; i<OUTPUT_SIZE;i++) {
        finals[i] = new ADV_InnerProduct(hidden_nodes, w.at(make_pair(1,i)));
    }

    ADV_Concat* ret = new ADV_Concat (finals);
    return finals[0];
  
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

int main() {
    cout << "Testing ADV_InnerProduct.\n";
    cout << "Single var tests.\n";
    cout << "Testing ADV_InnerProduct.\n";
    ADV_Vec x ("x", 1);
    ADV_InnerProduct f (&x,&x);
    f({{"x",{2}}}) ;
    f.take_gradient({1});
    double result = x.get_gradient()[0];
    cout << "result should be 4, it is: " << result << "\n";
    if(result != 4) return 1;
    f({{"x",{5}}}) ;
    f.take_gradient({1});
    result = x.get_gradient()[0];
    cout << "result should be 10, it is: " << result << "\n";
    if(result != 10) return 1;

    cout << "Testing ADV_Concat\n";
    ADV_Concat g ({&f});
    g( { {"x", {15}}});
    g.take_gradient({1});
    result = x.get_gradient()[0];
    cout << "result should be 30, it is: " << result << "\n";
    if(result != 30) return 1;

    ADV_Vec x_1 ("x_1", 2);
    ADV_Vec x_2 ("x_2", 2);
    ADV_InnerProduct x_s (&x_1, &x_2);
    x_s( { {"x_1", {1, 2}}, {"x_2", {3, 4}}});
    x_s.take_gradient({1});
    result = x_1.get_gradient()[0];
    cout << "result should be 3, it is: " << result << "\n";
    if(result != 3) return 1;

    ADV_Vec x11 ("x11", 1);
    ADV_Vec x12 ("x12", 1);
    ADV_Vec x21 ("x21", 1);
    ADV_Vec x22 ("x22", 1);

    ADV_Concat x1 ({&x11, &x12});
    ADV_Concat x2 ({&x21, &x22});
    ADV_InnerProduct xs (&x1, &x2);
    xs({ {"x11", {1}}
       , {"x12", {2}}
       , {"x21", {3}}
       , {"x22", {4}}
        });
    xs.take_gradient({1});
    result = x11.get_gradient()[0];
    cout << "result should be 3, it is : " << result << "\n";
    if(result != 3) return 1;

    ADV_Vec y11 ("y11", 1);
    ADV_Vec y12 ("y12", 1);
    ADV_Vec y21 ("y21", 1);
    ADV_Vec y22 ("y22", 1);

    ADV_Concat y1 ({&y11, &y12});
    ADV_Concat y2 ({&y21, &y22});

    ADV_InnerProduct ys (&y1, &y2);

    ADV_Concat xy ({&xs, &ys});
    ADV_InnerProduct concat_test (&xy, &xy);

    concat_test({ {"x11", {1}}
                , {"x12", {2}}
                , {"x21", {3}}
                , {"x22", {4}}
                , {"y11", {0}}
                , {"y12", {0}}
                , {"y21", {0}}
                , {"y22", {5}}
                 });
    concat_test.take_gradient({1});
    result = x11.get_gradient()[0];
    cout << "result should be 66, it is : " << result << "\n";
    if(result != 66) return 1;

    

    

    //cout << "Testing ADV_Negate\n";
    //ADV_Negate h (&x);
    //h( { {"x", {7}}});
    //h.take_gradient({1});
    //result = x.get_gradient()[0];
    //cout << "result should be -1, it is: " << result << "\n";
    //if(result != -1) return 1;

    //ADV_Negate h2 (&f);
    //h2( { {"x", {7}}});
    //h2.take_gradient({1});
    //result = x.get_gradient()[0];
    //cout << "result should be -14, it is: " << result << "\n";
    //if(result != -14) return 1;

    //cout << "Testing ADV_Sum\n";
    //ADV_Sum j (&f, &h); // x^2 - x
    //j( { { "x", {9}}});
    //j.take_gradient({1});
    //result = x.get_gradient()[0];
    //cout << "result should be 17, it is: " << result << "\n";
    //if(result != 17) return 1;

    //cout << "Testing ADV_LeakyReLU\n";
    //ADV_LeakyReLU k (&j);
    //k( { { "x", {3}}}); // LeakyReLU ( x^2 - x), d/dx = LeakyReLU'(x^2 - x) (2x - 1) 
    //k.take_gradient({1});
    //result = x.get_gradient()[0];
    //cout << "result should be 5, it is: " << result << "\n";
    //if(result != 5) return 1;

    //ADV_Negate nf (&f);
    //ADV_Sum snf(&x, &nf);
    //ADV_LeakyReLU l (&snf); // LeakyReLU( x-x^2), d/x = LeakyReLU'(x-x^2) (1-2x)
    //l({ { "x", {7}}});
    //l.take_gradient({1});
    //result = x.get_gradient()[0];
    //cout << "result should be -0.13, it is: " << result << "\n";
    //if(abs(result - -0.13) > 0.00001) return 1;
    //

    //cout << "Multivar tests.\n";
    //cout << "Testing ADV_InnerProduct.\n";
    //ADV_Vec y ("y", 3);
    //ADV_InnerProduct f1 (&y,&y);
    //f1({{"y",{2,3,5}}}) ;
    //f1.take_gradient({1});
    //valarray<double> res = y.get_gradient();
    //cout << "result should be {4,6,10}, it is: " << res << "\n";
    //if(res[0] != 4 || res[1] != 6 || res[2] != 10) return 1;

    //cout << "Testing ADV_LeakyReLU\n";
    //ADV_LeakyReLU f2 (&f1);
    //f2( { { "y", {1,2,3}}}); // LeakyReLU (y_1^2 + y_2^2 + y_3^2), d/dy_i = LeakyReLU'(\Sigma y_i^2)  2y_i
    //f2.take_gradient({1});
    //res = y.get_gradient();
    //cout << "result should be {2,4,6}}, it is: " << res << "\n";






    ADV_Weights weights {};

    int num_right_scratch = 0;
    int num_right_adv = 0;
    
    FFNN_Model model = initiate_weights_(INPUT_SIZE, NUM_HIDDEN_NODES, OUTPUT_SIZE, weights);
    ADV_Vec input ("input",INPUT_SIZE);
    ADV_Vec target ("target",OUTPUT_SIZE);
    ADV* predictor = get_predictor(weights, &input);
    ADV* error = get_error(predictor, &target);

    for(auto row : rows) {

        valarray<LayerOutput> model_out (model.output_size), output_0 (model.num_hidden_nodes);
        all_layer_0_outputs(model, row.x, output_0);
        model_output(model, output_0, model_out);
        double yp = (*predictor)({ {"input", row.x}})[0];

        if(row.y == lround(from_model_output(model_out))) num_right_scratch++;
        if(row.y == round(yp)) num_right_adv++;

        cout << "x: " << row.x << "\n"
             << "y: " << row.y << "\n"
             << "y-scratch: " << from_model_output(model_out) << "\n"
             << "y-adv: " << yp << "\n";

    }
    cout << "num right on scratch model: " << num_right_scratch << "/" << rows.size() << "\n";
    cout << "num right on adv model: " << num_right_adv << "/" << rows.size() << "\n";
    cout << "scratch model mean squared error: " << model_error(rows, model) << "\n";
    cout << "adv model mean squared error: " << model_error(rows, error) << "\n";

    double scratch_error = 99;
    double learning_rate = 0.1;
    int epochs = 1;
    int MAX_EPOCH = 2;
    num_right_adv=0;
    num_right_scratch=0;

    while(epochs <= MAX_EPOCH){
        for(auto row : rows) {

            valarray<double> scratch_gradients = train_weights(model, row, learning_rate);
            (*error)({ {"input", row.x}, {"target", row.t}});
            error->take_gradient({1});
            for(auto [coords, adv] : weights) {

                int start = coords.first == 0 ? coords.second * model.input_size : (model.input_size * model.num_hidden_nodes) + (coords.second * model.output_size);
                int size = coords.first == 0 ? model.input_size : model.num_hidden_nodes;
                valarray<double> scratch_grad = scratch_gradients[slice(start, size, 1)];

                valarray<double> adv_grad =  adv->get_gradient() ;


                    cout << "scratch gradient for weight (" << coords.first << "," << coords.second << "): "
                         << scratch_grad << "\n";
                    cout << "adv gradient for same: " << adv_grad << "\n";
                valarray<double> new_val = adv->val - (learning_rate * scratch_grad); //adv_grad);
                adv->setValue(new_val);
            }
        }
        scratch_error = model_error(rows, model);
        cout << "epoch " << epochs << " finished.\n";
        cout << "scratch model mean squared error:: " << scratch_error << "\n";
        cout << "adv model mean squared error: " << model_error(rows, error) << "\n";

        epochs++;
        for(auto row : rows) {

            valarray<LayerOutput> model_out (model.output_size), output_0 (model.num_hidden_nodes);
            all_layer_0_outputs(model, row.x, output_0);
            model_output(model, output_0, model_out);
            double yp = (*predictor)({ {"input", row.x}})[0];

            if(row.y == lround(from_model_output(model_out))) num_right_scratch++;
            if(row.y == round(yp)) num_right_adv++;

            cout << "x: " << row.x << "\n"
                 << "y: " << row.y << "\n"
                 << "y-scratch: " << from_model_output(model_out) << "\n"
                 << "y-adv: " << yp << "\n";

        }
        cout << "num right on scratch model: " << num_right_scratch << "/" << rows.size() << "\n";
        cout << "num right on adv model: " << num_right_adv << "/" << rows.size() << "\n";
        num_right_adv=0;
        num_right_scratch=0;
    }
    return 0;
}
