#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftLeakyReLUTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function *f;
    vector<vector<float>> result;
    vector<vector<float>> resultGrad;

    void SetUp() override {
        cublasCreate(&cublasH);

        f = new Function(&cublasH);
        f->addOp(Operation::matrix("z",2,2));
        f->addOp(Operation::applyLeakyReLU("f", "z", 2,2));
        f->compile();

    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftLeakyReLUTest, DaftLeakyReLUCompute) {
    f->setValue("z", {{ 500, -500, 0.5, -1 }});
    f->compute();
    f->computeGrad("f");
    vector<vector<float>> resultVec;
    f->getValue("f", &resultVec);
    f->getGrad("z", &resultGrad);

    float values[4] = { 500, -5, 0.5, -0.01 };
    float grads[4] = { 1, 0.01, 1, 0.01 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(resultVec[0][i], values[i]) << "LeakyReLU compute (" << i << ")";
        EXPECT_EQ(resultGrad[0][i], grads[i]) << "z grad (" << i << ")";
    }

}

class DaftBatchLeakyReLUTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add 3x2 matrix input and apply LeakyReLU
        f->addOp(Operation::column("input", 6)); 
        f->addOp(Operation::applyLeakyReLU("relu", "input", 6, 1));
        f->compile(5);  // Compile for batch size of 5
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchLeakyReLUTest, BatchLeakyReLUComputeAndGrad) {
    // Set 5 different 3x2 input matrices for batch processing
    vector<vector<float>> inputMatrices = {
        {2.0, -1.0, 0.5, -3.0, 1.5, -0.1},    // batch 0: mix of positive/negative
        {-2.0, -4.0, -1.0, -0.5, -10.0, -0.01}, // batch 1: all negative
        {5.0, 3.0, 1.0, 2.5, 0.1, 8.0},       // batch 2: all positive
        {0.0, -0.0, 1.0, -1.0, 100.0, -100.0}, // batch 3: zeros and extremes
        {0.25, -0.25, 4.0, -4.0, 0.75, -0.75}  // batch 4: small values
    };

    f->setValue("input", inputMatrices);
    f->compute();

    // Get results and verify LeakyReLU computation
    vector<vector<float>> results;
    f->getValue("relu", &results);

    // Expected results: positive values unchanged, negative values * 0.01
    vector<vector<float>> expected = {
        {2.0, -0.01, 0.5, -0.03, 1.5, -0.001},     // batch 0
        {-0.02, -0.04, -0.01, -0.005, -0.1, -0.0001}, // batch 1
        {5.0, 3.0, 1.0, 2.5, 0.1, 8.0},            // batch 2
        {0.0, 0.0, 1.0, -0.01, 100.0, -1.0},       // batch 3
        {0.25, -0.0025, 4.0, -0.04, 0.75, -0.0075} // batch 4
    };

    for (int batch = 0; batch < 5; batch++) {
        for (int elem = 0; elem < 6; elem++) {
            EXPECT_FLOAT_EQ(results[batch][elem], expected[batch][elem]) 
                << "Batch LeakyReLU compute batch " << batch << ", element " << elem;
        }
    }

    // Test gradients
    f->computeGrad("relu");
    vector<vector<float>> gradResults;
    f->getGrad("input", &gradResults);

    // Expected gradients: 1.0 for positive values, 0.01 for negative values
    vector<vector<float>> expectedGrads = {
        {1.0, 0.01, 1.0, 0.01, 1.0, 0.01},     // batch 0
        {0.01, 0.01, 0.01, 0.01, 0.01, 0.01},  // batch 1
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},        // batch 2
        {1.0, 1.0, 1.0, 0.01, 1.0, 0.01},      // batch 3
        {1.0, 0.01, 1.0, 0.01, 1.0, 0.01}      // batch 4
    };

    for (int batch = 0; batch < 5; batch++) {
        for (int elem = 0; elem < 6; elem++) {
            EXPECT_FLOAT_EQ(gradResults[batch][elem], expectedGrads[batch][elem]) 
                << "Batch LeakyReLU gradient batch " << batch << ", element " << elem;
        }
    }
}
