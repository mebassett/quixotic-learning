#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftBatchInnerProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add two 3-element column vectors for inner product
        f->addOp(Operation::column("vec1", 3));
        f->addOp(Operation::column("vec2", 3));
        
        // Compute inner product
        f->addOp(Operation::innerProduct("dotProduct", "vec1", "vec2", 3));
        f->compile(5);  // Compile for batch size of 5
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchInnerProductTest, BatchInnerProductComputeAndGradTest) {
    // Set 5 different 3D vectors for each input
    vector<vector<float>> vec1Inputs = {
        {1.0, 2.0, 3.0},     // batch 0: [1, 2, 3]
        {-1.0, 0.0, 1.0},    // batch 1: [-1, 0, 1]
        {2.5, -1.5, 0.5},    // batch 2: [2.5, -1.5, 0.5]
        {0.0, 0.0, 0.0},     // batch 3: zero vector
        {4.0, -2.0, 1.0}     // batch 4: [4, -2, 1]
    };

    vector<vector<float>> vec2Inputs = {
        {4.0, 5.0, 6.0},     // batch 0: [4, 5, 6]
        {2.0, 3.0, -1.0},    // batch 1: [2, 3, -1]
        {1.0, 2.0, 4.0},     // batch 2: [1, 2, 4]
        {1.0, 1.0, 1.0},     // batch 3: ones vector
        {0.5, 1.5, -0.5}     // batch 4: [0.5, 1.5, -0.5]
    };

    f->setValue("vec1", vec1Inputs);
    f->setValue("vec2", vec2Inputs);
    f->compute();

    // Get results and verify inner products
    vector<vector<float>> results;
    f->getValue("dotProduct", &results);

    // Expected results: vec1[i] · vec2[i] for each batch i
    vector<float> expected = {
        1.0*4.0 + 2.0*5.0 + 3.0*6.0,        // batch 0: 4 + 10 + 18 = 32
        -1.0*2.0 + 0.0*3.0 + 1.0*(-1.0),    // batch 1: -2 + 0 + (-1) = -3
        2.5*1.0 + (-1.5)*2.0 + 0.5*4.0,     // batch 2: 2.5 - 3 + 2 = 1.5
        0.0*1.0 + 0.0*1.0 + 0.0*1.0,        // batch 3: 0 + 0 + 0 = 0
        4.0*0.5 + (-2.0)*1.5 + 1.0*(-0.5)   // batch 4: 2 - 3 - 0.5 = -1.5
    };

    for (int batch = 0; batch < 5; batch++) {
        EXPECT_FLOAT_EQ(results[batch][0], expected[batch]) 
            << "Batch inner product compute batch " << batch;
    }

    // Test gradients
    f->computeGrad("dotProduct");
    
    vector<vector<float>> vec1Grads;
    vector<vector<float>> vec2Grads;
    f->getGrad("vec1", &vec1Grads);
    f->getGrad("vec2", &vec2Grads);

    // For inner product, gradient of vec1 should be vec2, and gradient of vec2 should be vec1
    for (int batch = 0; batch < 5; batch++) {
        for (int elem = 0; elem < 3; elem++) {
            EXPECT_FLOAT_EQ(vec1Grads[batch][elem], vec2Inputs[batch][elem]) 
                << "Batch inner product vec1 gradient batch " << batch << ", element " << elem;
            EXPECT_FLOAT_EQ(vec2Grads[batch][elem], vec1Inputs[batch][elem]) 
                << "Batch inner product vec2 gradient batch " << batch << ", element " << elem;
        }
    }
}
