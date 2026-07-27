#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>
#include <map>
#include <string>

using namespace DA;

class DaftConcatComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> result;
    vector<vector<float>> testgrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add two single-element columns and one 2-element column
        f->addOp(Operation::column("v2", 1));
        f->addOp(Operation::column("v3", 1));
        f->addOp(Operation::column("v4", 2));
        
        // Add concat operation to combine v2 and v3 into a 2-element vector
        f->addOp(Operation::concat("v2v3", {"v2", "v3"}, 2));
        
        // Add inner product between the concatenated vector and v4
        f->addOp(Operation::innerProduct("flatF", "v2v3", "v4", 2));
        
        f->compile();

        // Set up values - now using vector of vectors for batch API
        f->setValue("v2", {{1}});
        f->setValue("v3", {{2}});
        f->setValue("v4", {{3, 5}});

        f->compute();
        f->computeGrad("flatF");
        
        f->getValue("flatF", &result);
        f->getGrad("v2", &testgrad);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftConcatComputeTest, ConcatComputeTest) {
    EXPECT_EQ(result[0][0], 13) << "Daft Concat compute";
    EXPECT_EQ(testgrad[0][0], 3) << "Daft Concat grad";
}

class DaftBatchConcatComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> result;
    vector<vector<float>> testgrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add two single-element columns and one 2-element column
        f->addOp(Operation::column("v2", 1));
        f->addOp(Operation::column("v3", 1));
        f->addOp(Operation::column("v4", 2));

        // Concatenate v2 and v3 into a 2-element vector
        f->addOp(Operation::concat("v2v3", {"v2", "v3"}, 2));

        // Inner product of the concatenated vector with v4
        f->addOp(Operation::innerProduct("flatF", "v2v3", "v4", 2));

        f->compile(4);  // Compile for batch size of 4

        // Set batch values for 4 independent examples
        // v2 = {1, 2, 3, 4}
        f->setValue("v2", {{1}, {2}, {3}, {4}});
        // v3 = {2, 3, 4, 5}
        f->setValue("v3", {{2}, {3}, {4}, {5}});
        // v4 = {{3,5}, {1,2}, {4,6}, {2,3}}
        f->setValue("v4", {{3, 5}, {1, 2}, {4, 6}, {2, 3}});

        f->compute();
        f->computeGrad("flatF");

        f->getValue("flatF", &result);
        f->getGrad("v2", &testgrad);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchConcatComputeTest, BatchConcatComputeTest) {
    // Expected: v2v3[i] = {v2[i], v3[i]} then flatF[i] = v2v3[i] · v4[i]
    // batch 0: {1,2} · {3,5} = 1*3 + 2*5 = 13
    // batch 1: {2,3} · {1,2} = 2*1 + 3*2 = 8
    // batch 2: {3,4} · {4,6} = 3*4 + 4*6 = 36
    // batch 3: {4,5} · {2,3} = 4*2 + 5*3 = 23
    float expectedResults[4] = {13, 8, 36, 23};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(result[i][0], expectedResults[i])
            << "Batch concat compute batch " << i;
    }

    // Gradient of v2 = v4[0] for each batch
    // batch 0: 3, batch 1: 1, batch 2: 4, batch 3: 2
    float expectedGrads[4] = {3, 1, 4, 2};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(testgrad[i][0], expectedGrads[i])
            << "Batch concat gradient batch " << i;
    }
}
