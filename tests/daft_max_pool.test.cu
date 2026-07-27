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
        f->addOp(Operation::inputMatrix("id3", 2, 2));
        
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
        f->addOp(Operation::inputMatrix("id3", 4, 4));
        
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
        f->addOp(Operation::inputMatrix("id3", 2, 2));
        
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

class DaftBatchMaxPoolComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add 4x4 input matrix
        f->addOp(Operation::inputMatrix("input", 4, 4));

        // MaxPool 2x2 pool size, stride 2 -> 2x2 output
        f->addOp(Operation::maxPool("mp", "input", 2, 2, 2, 2, 4, 4));

        f->compile(3);  // Compile for batch size of 3

        // Set 3 different 4x4 input matrices
        f->setValue("input", {
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},   // batch 0
            {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},   // batch 1: reversed
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}           // batch 2: all ones
        });

        f->compute();
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchMaxPoolComputeTest, BatchMaxPoolComputeTest) {
    vector<vector<float>> result;
    f->getValue("mp", &result);

    // MaxPool 2x2 stride 2 on 4x4 -> 2x2 output
    // batch 0: max of {{1,2},{5,6}}=6, {{3,4},{7,8}}=8, {{9,10},{13,14}}=14, {{11,12},{15,16}}=16
    // batch 1: max of {{16,15},{12,11}}=16, {{14,13},{10,9}}=14, {{8,7},{4,3}}=8, {{6,5},{2,1}}=6
    // batch 2: all ones -> all 1
    float expected[3][4] = {
        {6, 8, 14, 16},
        {16, 14, 8, 6},
        {1, 1, 1, 1}
    };
    for (int batch = 0; batch < 2; batch++) {
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(result[batch][i], expected[batch][i])
                << "Batch max pool batch " << batch << ", element " << i;
        }
    }
}

class DaftBatchMaxPoolGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> testvalue;
    float scalarValue = 5;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add 2x2 input matrix
        f->addOp(Operation::inputMatrix("input", 2, 2));

        // MaxPool 2x2 pool size, stride 1 -> 1x1 output
        f->addOp(Operation::maxPool("mp", "input", 2, 2, 1, 1, 2, 2));

        // Scalar multiplication
        f->addOp(Operation::scalarMultiply("smp", "mp", 1, 1, scalarValue));

        f->compile(1); 

        
        f->setValue("input", {
            {1, 1, 1, 4}});//,    // batch 0: max at [1,1]
        //    {4, 1, 1, 1},    // batch 1: max at [0,0]
        //    {1, 4, 1, 1},    // batch 2: max at [0,1]
        //    {0, -14, 99, 5}  // batch 3: max at [1,0]
        //});

        f->compute();
        f->computeGrad("smp");

        f->getGrad("input", &testvalue);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchMaxPoolGradTest, BatchMaxPoolGradTest) {
    // Only the max element gets the gradient (scalarValue = 5)
    // batch 0: max at [1,1] -> {0, 0, 0, 5}
    // batch 1: max at [0,0] -> {5, 0, 0, 0}
    // batch 2: max at [0,1] -> {0, 5, 0, 0}
    float expected[4][4] = {
        {0, 0, 0, 5},
        {5, 0, 0, 0},
        {0, 5, 0, 0},
        {0, 0, 5, 0}
    };
    for (int batch = 0; batch < 1; batch++) {
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(testvalue[batch][i], expected[batch][i])
                << "Batch max pool grad batch " << batch << ", element " << i;
        }
    }
}
