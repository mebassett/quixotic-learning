#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
#include<valarray>
import fast_autodiff;
import mnistdata;
    
using namespace std;
const unsigned int INPUT_SIZE = 784;
const unsigned int NUM_HIDDEN_NODES= 100;
const unsigned int OUTPUT_SIZE = 10;

void initialize_weights(Matrix* m1, Matrix* m2, double lower_bound, double upper_bound) {
    random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    auto get_rand = [&](){ return dis(rd); } ;

    for(int i {0}; i < m1->rows * m1->cols; i++)
        *(m1->value+i) = get_rand();
    
    for(int i {0}; i < m2->rows * m2->cols; i++)
        *(m2->value+i) = get_rand();
}

int main() {
     Matrix* w1 = new Matrix ("weight1", NUM_HIDDEN_NODES, INPUT_SIZE);
     Matrix* w2 = new Matrix ("weight2", OUTPUT_SIZE, NUM_HIDDEN_NODES);

     initialize_weights(w1, w2, -0.05, 0.05);

     Col* feature_input = new Col("feature_input", INPUT_SIZE);
     Col* target_input = new Col("target_input", OUTPUT_SIZE);

     MatrixColProduct* predictor = new MatrixColProduct (w2, new ColLeakyReLU(new MatrixColProduct(w1, feature_input)));

    AddCol* term_error = new AddCol(target_input, new Scalar(predictor, -1.0));

    Scalar* error_func = new Scalar(new InnerProduct(term_error, term_error), 0.5); 
    
    Training_Data rows = load_data_from_file("mnist_train.txt", 60000);
    Training_Data test_rows = load_data_from_file("mnist_test.txt", 10000);
     


}
