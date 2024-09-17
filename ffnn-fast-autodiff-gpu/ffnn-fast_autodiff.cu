#include <iostream>
#include <vector>
#include<map>
#include<utility>
#include<random>
#include<valarray>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "fast_autodiff.h"
#include "mnistdata.h"
    
using namespace std;
using namespace FA;
using namespace MNIST;

const unsigned int NUM_HIDDEN_NODES= 100;

void initialize_weights(Matrix* m1, Matrix* m2, float lower_bound, float upper_bound) {
    random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    auto get_rand = [&](){ return dis(rd); } ;

    valarray<float> m1Weights (m1->rows * m1->cols);

    for(int i {0}; i < m1->rows * m1->cols; i++)
        m1Weights[i] = get_rand();
    m1->loadValues(m1Weights);
    
    valarray<float> m2Weights (m2->rows * m2->cols);
    for(int i {0}; i < m2->rows * m2->cols; i++)
        m2Weights[i] = get_rand();
    m2->loadValues(m2Weights);
}

int fromModelOutput(float* out) {
  float max = *max_element(out, out + OUTPUT_SIZE);
  for(int i {0}; i<OUTPUT_SIZE; i++) {
    if(*(out+i) == max) return i;
  }
  return -1;
}

int main() {
     float learningRate = 0.025;
     cublasHandle_t cublasH;

     cublasCreate(&cublasH);


     Matrix* w1 = new Matrix ("weight1", NUM_HIDDEN_NODES, INPUT_SIZE + 1);
     Matrix* w2 = new Matrix ("weight2", OUTPUT_SIZE, NUM_HIDDEN_NODES);

     cout << "init weights...\n";

     initialize_weights(w1, w2, -0.05, 0.05);

     Col* featureInput = new Col("featureInput", INPUT_SIZE + 1);
     Col* targetInput = new Col("targetInput", OUTPUT_SIZE);

     MatrixColProduct* predictor = new MatrixColProduct (w2, new ColLeakyReLU(new MatrixColProduct(w1, featureInput)));

    AddCol* termError = new AddCol(targetInput, new Scalar(predictor, -1.0));

    Scalar* errorFunc = new Scalar(new InnerProduct(termError, termError), 0.5); 
    
    Training_Data rows = load_data_from_file("../data/mnist_train.txt", 60000);
    Training_Data testRows = load_data_from_file("../data/mnist_test.txt", 10000);
     
    int count = 1;
    int numRight = 0;
    int trainingExamples = 0;
    float errorRate;

    numRight = 0;
    errorRate = 0.0;
    trainingExamples = 0;
    for (auto row: testRows){
        featureInput->loadValues(row.x);
        targetInput->loadValues(row.t);

        errorFunc->compute(&cublasH);
        errorFunc->fromDevice();
        predictor->fromDevice();

        int out = fromModelOutput(predictor->value);

        errorRate += *(errorFunc->value);
        if(out == row.y) numRight++;
    }
    cout << "num right: " << numRight << " / " << testRows.size() << " .\n";
    cout << "model error on test set:" << errorRate << " .\n";

    while(count <= 1) {

        cout << "starting epoch " << count << ".\n";
        for (auto row: rows) {
            errorFunc->resetGrad();
            
            featureInput->loadValues(row.x);
            targetInput->loadValues(row.t);

            errorFunc->compute(&cublasH);
            errorFunc->computeGrad(&cublasH);

            w1->gradDescent(&cublasH, learningRate);
            w2->gradDescent(&cublasH, learningRate);


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

            errorFunc->compute(&cublasH);
            errorFunc->fromDevice();
            predictor->fromDevice();

            int out = fromModelOutput(predictor->value);

            errorRate += *(errorFunc->value);
            if(out == row.y) numRight++;
        }
        cout << "num right: " << numRight << " / " << testRows.size() << " .\n";
        cout << "model error on test set:" << errorRate << " .\n";

        count++;

    }

    delete errorFunc;

}
