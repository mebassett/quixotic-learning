#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftBatchScalarTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::scalarMultiply("test", "xy", 2, 1, 5.0));
        f->compile(4);  // Compile for batch size of 4
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchScalarTest, BatchScalarComputeAndGradTest) {
    // Set 4 different 2D vectors for batch processing
    vector<vector<float>> inputVectors = {
        {1.0, 2.0},     // batch 0: [1, 2]
        {-0.5, 3.5},    // batch 1: [-0.5, 3.5]
        {0.0, -1.0},    // batch 2: [0, -1]
        {2.5, -2.5}     // batch 3: [2.5, -2.5]
    };

    f->setValue("xy", inputVectors);
    f->compute();

    // Get results and verify scalar multiplication
    vector<vector<float>> results;
    f->getValue("test", &results);

    // Expected results: each input vector multiplied by 5.0
    vector<vector<float>> expected = {
        {5.0, 10.0},     // batch 0: [1*5, 2*5]
        {-2.5, 17.5},    // batch 1: [-0.5*5, 3.5*5]
        {0.0, -5.0},     // batch 2: [0*5, -1*5]
        {12.5, -12.5}    // batch 3: [2.5*5, -2.5*5]
    };

    for (int batch = 0; batch < 4; batch++) {
        for (int elem = 0; elem < 2; elem++) {
            EXPECT_FLOAT_EQ(results[batch][elem], expected[batch][elem]) 
                << "Batch scalar compute batch " << batch << ", element " << elem;
        }
    }

    // Test gradients
    f->computeGrad("test");
    vector<vector<float>> gradResults;
    f->getGrad("xy", &gradResults);

    // Expected gradients: all should be 5.0 (the scalar multiplier)
    for (int batch = 0; batch < 4; batch++) {
        for (int elem = 0; elem < 2; elem++) {
            EXPECT_FLOAT_EQ(gradResults[batch][elem], 5.0) 
                << "Batch scalar gradient batch " << batch << ", element " << elem;
        }
    }
}
