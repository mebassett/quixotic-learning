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
