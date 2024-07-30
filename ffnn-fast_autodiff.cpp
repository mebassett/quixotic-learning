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

int fromModelOutput(double* out) {
  double max = *max_element(out, out + OUTPUT_SIZE);
  for(int i {0}; i<OUTPUT_SIZE; i++) {
    if(*(out+i) == max) return i;
  }
  return -1;
}

int main() {
     double learningRate = 0.025;

     Matrix* w1 = new Matrix ("weight1", NUM_HIDDEN_NODES, INPUT_SIZE);
     Matrix* w2 = new Matrix ("weight2", OUTPUT_SIZE, NUM_HIDDEN_NODES);

     initialize_weights(w1, w2, -0.05, 0.05);

     Col* featureInput = new Col("featureInput", INPUT_SIZE + 1);
     Col* targetInput = new Col("targetInput", OUTPUT_SIZE);

     MatrixColProduct* predictor = new MatrixColProduct (w2, new ColLeakyReLU(new MatrixColProduct(w1, featureInput)));

    AddCol* termError = new AddCol(targetInput, new Scalar(predictor, -1.0));

    Scalar* errorFunc = new Scalar(new InnerProduct(termError, termError), 0.5); 
    
    Training_Data rows = load_data_from_file("mnist_train.txt", 60000);
    Training_Data testRows = load_data_from_file("mnist_test.txt", 10000);
     
    int count = 1;
    int numRight = 0;
    int trainingExamples = 0;
    double errorRate;

    double seed[1] = { 1.0 };

    while(count <= 1) {

        cout << "starting epoch " << count << ".\n";
        for (auto row: rows) {
            errorFunc->resetGrad();
            
            featureInput->loadValues(row.x);
            targetInput->loadValues(row.t);

            errorFunc->compute();
            errorFunc->pushGrad(seed);

            for(int i {0}; i < w1->rows * w1->cols; i++)
                *(w1->value + i) = *(w1->value + i) - (learningRate * *(w1->grad + i));

            for(int i {0}; i < w2->rows * w2->cols; i++)
                *(w2->value + i) = *(w2->value + i) - (learningRate * *(w2->grad + i));

            trainingExamples++;
            if(trainingExamples % 10000 == 0)
                cout << "done " << trainingExamples << " so far.\n";

        }
        numRight = 0;
        errorRate = 0.0;
        trainingExamples = 0;
        for (auto row: testRows){
            featureInput->loadValues(row.x);
            targetInput->loadValues(row.t);

            errorFunc->compute();

            int out = fromModelOutput(predictor->value);
            errorRate += *(errorFunc->value);
            if(out == row.y) numRight++;
        }
        cout << "num right: " << numRight << " / " << testRows.size() << " .\n";
        cout << "model error on test set:" << errorRate << " .\n";

        count++;

    }

}
