#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftMaxPoolComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add 2x2 input matrix
        f->addOp(Operation::matrix("id3", 2, 2));
        
        // Add MaxPool operation with 2x2 pool size and stride 1
        f->addOp(Operation::maxPool("mp", "id3", 2, 2, 1, 1, 2, 2));
        
        f->compile(1);  // Add batch size of 1

        // Set up values - same as silly test
        f->setValue("id3", {{1, 2, 3, 4}});

        f->compute();
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftMaxPoolComputeTest, MaxPoolComputeTest) {
    vector<vector<float>> result;
    f->getValue("mp", &result);
    EXPECT_EQ(result[0][0], 4) << "Daft MaxPool compute";
}

class DaftMaxPoolLargeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add 4x4 input matrix
        f->addOp(Operation::matrix("id3", 4, 4));
        
        // Add MaxPool operation with 2x2 pool size and stride 2
        f->addOp(Operation::maxPool("mp", "id3", 2, 2, 2, 2, 4, 4));
        
        f->compile(1);  // Add batch size of 1

        // Set up values - same as silly test
        f->setValue("id3", {{1, 2, 1, 2, 3, 9, 16, 3, 1, 10, 4, 1, 3, 4, 2, 3}});

        f->compute();
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftMaxPoolLargeTest, MaxPoolLargeTest) {
    vector<vector<float>> result;
    f->getValue("mp", &result);
    float values[4] = {9, 16, 10, 4};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(result[0][i], values[i]) << "Daft MaxPool large test";
    }
}

class DaftMaxPoolGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> testvalue;
    float scalarValue = 5;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add 2x2 input matrix
        f->addOp(Operation::matrix("id3", 2, 2));
        
        // Add MaxPool operation with 2x2 pool size and stride 1
        f->addOp(Operation::maxPool("mp", "id3", 2, 2, 1, 1, 2, 2));
        
        // Add scalar multiplication
        f->addOp(Operation::scalarMultiply("smp", "mp", 1, 1, scalarValue));
        
        f->compile(1);  // Add batch size of 1

        // Set up values - same as silly test
        f->setValue("id3", {{1, 1, 1, 4}});

        f->compute();
        f->computeGrad("smp");
        
        f->getGrad("id3", &testvalue);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftMaxPoolGradTest, MaxPoolGradTest) {
    float values[4] = {0, 0, 0, scalarValue};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(testvalue[0][i], values[i]) << "Daft MaxPool grad test";
    }
}
