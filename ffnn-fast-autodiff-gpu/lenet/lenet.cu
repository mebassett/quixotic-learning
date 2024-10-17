#include <chrono>
#include <iterator>
#include <random>
#include <valarray>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../fast_autodiff/fast_autodiff.h"
#include "../mnistdata/mnistdata.h"

using namespace std;
using namespace FA;
using namespace MNIST;

void initialize_weights(Matrix* m1, float lower_bound, float upper_bound)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    auto get_rand = [&]() { return dis(rd); };

    valarray<float> m1Weights(m1->rows * m1->cols);

    for (int i { 0 }; i < m1->rows * m1->cols; i++)
        m1Weights[i] = get_rand();
    m1->loadValues(m1Weights);
}
int fromModelOutput(float* out)
{
    float max = *max_element(out, out + OUTPUT_SIZE);
    for (int i { 0 }; i < OUTPUT_SIZE; i++) {
        if (*(out + i) == max)
            return i;
    }
    return -1;
}

int main()
{
    // see https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf for architecture.
    cout << "Building LeNet...\n";

    Matrix* featureInput = new Matrix("featureInput", 28, 28);
    Col* targetInput = new Col("targetInput", 10);

    Matrix* c1LayerWeights[6];
    Convolution* c1Layer[6];
    MaxPool* s2Layer[6];
    for (int i = 0; i < 6; i++) {
        c1LayerWeights[i] = new Matrix("c1-kerenl-" + to_string(i), 5, 5);
        c1Layer[i] = new Convolution(featureInput, c1LayerWeights[i], 2, 1, 2, 1);
        s2Layer[i] = new MaxPool(new ColLeakyReLU(c1Layer[i]), 2, 2, 2, 2);
    }

    Matrix* c3LayerWeights[16];
    Add* c3Layer[16];
    for (int i = 0; i < 16; i++) {
        c3LayerWeights[i] = new Matrix("c3-kernel-" + to_string(i), 5, 5);
    }

    // we only do 2d tensors (matricies), so we treat each channel individually.
    // the first 6 channels are convolutions of a subset of 3 continuous channels
    // in the previous layer.  convolution is just a dot product, so we can add
    // the convoluted channels together.
    int c3TripleIndex[6][3] = { { 0, 1, 2 }, { 1, 2, 3 }, { 2, 3, 4 },
        { 3, 4, 5 }, { 4, 5, 0 }, { 5, 0, 1 } };
    for (int i = 0; i < 6; i++) {
        int r = c3TripleIndex[i][0];
        int s = c3TripleIndex[i][1];
        int t = c3TripleIndex[i][2];
        c3Layer[i] = new Add(
            new Convolution(s2Layer[r], c3LayerWeights[i], 0, 1, 0, 1),
            new Add(new Convolution(s2Layer[s], c3LayerWeights[i], 0, 1, 0, 1),
                new Convolution(s2Layer[t], c3LayerWeights[i], 0, 1, 0, 1)));
    }

    // the next 6 channels are convolutions of a subset of 4 continous channels.
    // and then we have 3 convolutions of 4 discontinuous channels
    int c3QuadIndex[9][4] = { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 },
        { 3, 4, 5, 0 }, { 4, 5, 0, 1 }, { 5, 0, 1, 2 },
        { 0, 1, 3, 4 }, { 1, 2, 4, 5 }, { 0, 2, 3, 5 } };
    for (int i = 6; i < 15; i++) {
        int r = c3QuadIndex[i - 6][0];
        int s = c3QuadIndex[i - 6][1];
        int t = c3QuadIndex[i - 6][2];
        int u = c3QuadIndex[i - 6][3];
        c3Layer[i] = new Add(
            new Convolution(s2Layer[r], c3LayerWeights[i], 0, 1, 0, 1),
            new Add(
                new Convolution(s2Layer[s], c3LayerWeights[i], 0, 1, 0, 1),
                new Add(
                    new Convolution(s2Layer[t], c3LayerWeights[i], 0, 1, 0, 1),
                    new Convolution(s2Layer[u], c3LayerWeights[i], 0, 1, 0, 1))));
    }
    // the last channel is a convolution on all 6 prior channels.
    c3Layer[15] = new Add(
        new Convolution(s2Layer[0], c3LayerWeights[15], 0, 1, 0, 1),
        new Add(
            new Convolution(s2Layer[1], c3LayerWeights[15], 0, 1, 0, 1),
            new Add(
                new Convolution(s2Layer[2], c3LayerWeights[15], 0, 1, 0, 1),
                new Add(
                    new Convolution(s2Layer[3], c3LayerWeights[15], 0, 1, 0, 1),
                    new Add(new Convolution(s2Layer[4], c3LayerWeights[15], 0, 1,
                                0, 1),
                        new Convolution(s2Layer[0], c3LayerWeights[15], 0, 1,
                            0, 1))))));

    AD* s4Layer[16];
    for (int i = 0; i < 16; i++) {
        s4Layer[i] = new Flatten(new MaxPool(new ColLeakyReLU(c3Layer[i]), 2, 2, 2, 2));
    }

    ConcatCol* s4 = new ConcatCol(vector(begin(s4Layer), end(s4Layer)));

    Matrix* wc5 = new Matrix("fully-connected-weight-0", 120, 400);
    ColLeakyReLU* c5 = new ColLeakyReLU(new MatrixColProduct(wc5, s4));

    Matrix* wf6 = new Matrix("fully-connected-weight-1", 84, 120);
    ColLeakyReLU* f6 = new ColLeakyReLU(new MatrixColProduct(wf6, c5));

    Matrix* wOutput = new Matrix("fully-connected-weight-2", 10, 84);
    MatrixColProduct* leNetOut = new MatrixColProduct(wOutput, f6);

    Add* termError = new Add(targetInput, new Scalar(leNetOut, -1.0));
    Scalar* lossFunc = new Scalar(new InnerProduct(termError, termError), 0.5);

    cout << "Model built, now initializing weights...\n";
    for_each(begin(c1LayerWeights), end(c1LayerWeights),
        [](auto m) { initialize_weights(m, -0.05, 0.05); });
    for_each(begin(c3LayerWeights), end(c3LayerWeights),
        [](auto m) { initialize_weights(m, -0.05, 0.05); });
    initialize_weights(wc5, -0.05, 0.05);
    initialize_weights(wf6, -0.05, 0.05);
    initialize_weights(wOutput, -0.05, 0.05);

    cout << "Now we are getting the initial error of the model...\n";
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    Training_Data rows = load_data_from_file_no_bias("../data/mnist_train.txt", 60000);
    Training_Data testRows = load_data_from_file_no_bias("../data/mnist_test.txt", 10000);

    int numRight = 0;
    float errorRate;

    numRight = 0;
    errorRate = 0.0;
    for (auto row : testRows) {
        featureInput->loadValues(row.x);
        targetInput->loadValues(row.t);

        lossFunc->compute(&cublasH);
        lossFunc->fromDevice();
        leNetOut->fromDevice();

        int out = fromModelOutput(leNetOut->value);

        errorRate += *(lossFunc->value);
        if (out == row.y)
            numRight++;
    }
    cout << "num right: " << numRight << " / " << testRows.size() << " .\n";
    cout << "model error on test set:" << errorRate << " .\n";

    cout << "\nTraining!\n\n";

    int epochs = 1;
    float learningRate = 0.025;
    int trainingExamples = 0;
    auto startTime = chrono::steady_clock::now();
    while (epochs <= 20) {
        auto epochStartTime = chrono::steady_clock::now();
        cout << "Starting epoch " << epochs << "...\n";
        for (auto row : rows) {
            lossFunc->resetGrad();

            featureInput->loadValues(row.x);
            targetInput->loadValues(row.t);

            lossFunc->compute(&cublasH);
            lossFunc->computeGrad(&cublasH);
            for (auto m : c1LayerWeights) {
                m->gradDescent(&cublasH, learningRate);
            }
            for (auto m : c3LayerWeights) {
                m->gradDescent(&cublasH, learningRate);
            }

            wc5->gradDescent(&cublasH, learningRate);
            wf6->gradDescent(&cublasH, learningRate);
            wOutput->gradDescent(&cublasH, learningRate);

            trainingExamples++;
            if (trainingExamples % 10000 == 0)
                cout << "done " << trainingExamples << " examples so far.\n";
        }
        auto epochEndTime = chrono::steady_clock::now();
        auto elapsed = epochEndTime - startTime;
        auto lapped = epochEndTime - epochStartTime;
        trainingExamples = 0;
        cout << "training finished, testing again\n";
        cout << "Elapsed time: "
             << chrono::duration_cast<chrono::seconds>(elapsed).count() << " s\n";
        cout << "Epoch time: "
             << chrono::duration_cast<chrono::seconds>(lapped).count() << " s\n";
        numRight = 0;
        errorRate = 0.0;
        for (auto row : testRows) {
            featureInput->loadValues(row.x);
            targetInput->loadValues(row.t);

            lossFunc->compute(&cublasH);
            lossFunc->fromDevice();
            leNetOut->fromDevice();

            int out = fromModelOutput(leNetOut->value);

            errorRate += *(lossFunc->value);
            if (out == row.y)
                numRight++;
        }
        cout << "num right: " << numRight << " / " << testRows.size() << " .\n";
        cout << "model error on test set:" << errorRate << " .\n\n";

        epochs++;
    }

    return 0;
}
