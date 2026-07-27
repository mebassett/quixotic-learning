#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftBatchNegativeIdentityRegularTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add 2x2 negative identity matrix and 2x1 column vector
        f->addOp(Operation::weightsMatrix("negId", 2, 2));
        f->addOp(Operation::column("x", 2));
        
        // Multiply negative identity by vector
        f->addOp(Operation::matrixProduct("result", "negId", "x", 2, 2, 1));
        f->compile(10);  // Compile for batch size of 10

        // Set negative identity matrix: [[-1, 0], [0, -1]]
        // Note: InputMatrix only needs one copy since it's special
        f->setValue("negId", {{-1, 0, 0, -1}});
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchNegativeIdentityRegularTest, BatchNegativeIdentityRegularTest) {
    // Set 10 different 2D vectors for batch processing
    vector<vector<float>> inputVectors = {
        {1.5, 2.3},    // batch 0
        {-0.7, 4.1},   // batch 1
        {3.2, -1.8},   // batch 2
        {0.0, 5.5},    // batch 3
        {-2.4, -3.1},  // batch 4
        {7.7, 0.9},    // batch 5
        {-5.0, 2.0},   // batch 6
        {1.1, -4.4},   // batch 7
        {6.6, 8.8},    // batch 8
        {-9.9, -0.1}   // batch 9
    };

    f->setValue("x", inputVectors);
    f->compute();

    // Get results and verify each vector becomes its negative
    vector<vector<float>> results;
    f->getValue("result", &results);

    vector<vector<float>> expected = {
        {-1.5, -2.3},    // batch 0
        {0.7, -4.1},     // batch 1
        {-3.2, 1.8},     // batch 2
        {0.0, -5.5},     // batch 3
        {2.4, 3.1},      // batch 4
        {-7.7, -0.9},    // batch 5
        {5.0, -2.0},     // batch 6
        {-1.1, 4.4},     // batch 7
        {-6.6, -8.8},    // batch 8
        {9.9, 0.1}       // batch 9
    };

    for (int batch = 0; batch < 10; batch++) {
        EXPECT_FLOAT_EQ(results[batch][0], expected[batch][0]) 
            << "Regular batch negative identity batch " << batch << ", element 0";
        EXPECT_FLOAT_EQ(results[batch][1], expected[batch][1]) 
            << "Regular batch negative identity batch " << batch << ", element 1";
    }
}
