#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftBatchAddTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::add("f", "xy", "xy", 2, 1));
        f->compile(6);  // Compile for batch size of 6
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchAddTest, BatchAddComputeAndGradTest) {
    // Set 6 different 2D vectors for batch processing
    vector<vector<float>> inputVectors = {
        {1.0, 2.0},      // batch 0: [1, 2]
        {-3.0, 4.5},     // batch 1: [-3, 4.5]
        {0.0, -1.0},     // batch 2: [0, -1]
        {2.5, -2.5},     // batch 3: [2.5, -2.5]
        {10.0, 0.1},     // batch 4: [10, 0.1]
        {-0.5, 7.0}      // batch 5: [-0.5, 7]
    };

    f->setValue("xy", inputVectors);
    f->compute();

    // Get results and verify addition (xy + xy = 2*xy)
    vector<vector<float>> results;
    f->getValue("f", &results);

    // Expected results: each input vector added to itself (doubled)
    vector<vector<float>> expected = {
        {2.0, 4.0},      // batch 0: [1*2, 2*2]
        {-6.0, 9.0},     // batch 1: [-3*2, 4.5*2]
        {0.0, -2.0},     // batch 2: [0*2, -1*2]
        {5.0, -5.0},     // batch 3: [2.5*2, -2.5*2]
        {20.0, 0.2},     // batch 4: [10*2, 0.1*2]
        {-1.0, 14.0}     // batch 5: [-0.5*2, 7*2]
    };

    for (int batch = 0; batch < 6; batch++) {
        for (int elem = 0; elem < 2; elem++) {
            EXPECT_FLOAT_EQ(results[batch][elem], expected[batch][elem]) 
                << "Batch add compute batch " << batch << ", element " << elem;
        }
    }

    // Test gradients
    f->computeGrad("f");
    vector<vector<float>> gradResults;
    f->getGrad("xy", &gradResults);

    // Expected gradients: for addition f = xy + xy, gradient of xy should be 2.0
    // (since df/d(xy) = 1 + 1 = 2)
    for (int batch = 0; batch < 6; batch++) {
        for (int elem = 0; elem < 2; elem++) {
            EXPECT_FLOAT_EQ(gradResults[batch][elem], 2.0) 
                << "Batch add gradient batch " << batch << ", element " << elem;
        }
    }
}
