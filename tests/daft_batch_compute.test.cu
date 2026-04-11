#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>
#include <map>
#include <string>

using namespace DA;

class DaftBatchComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    map<string, vector<vector<float>>*> results;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        f->addOp(Operation::column("a", 2));
        f->addOp(Operation::column("b", 2));
        f->addOp(Operation::innerProduct("result1","a","b",2));
        f->addOp(Operation::innerProduct("result2", "result1", "result1", 1));
        f->compile();

        results["result1"] = new vector<vector<float>>;
        results["result2"] = new vector<vector<float>>;

        map<string, vector<vector<float>>> inputs
         {{"a", { { 1, 2}, {0,0}, {3, 4}}}, {"b", { { 0, 1}, {9,9}, {1,0}}}};

        f->batchCompute(results, {"result1","result2"}, inputs); 
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete results["result1"];
        delete results["result2"];
        delete f;
    }
};

TEST_F(DaftBatchComputeTest, BatchComputeTest) {
    EXPECT_EQ((*results["result1"])[0][0], 2) << "result1-0";
    EXPECT_EQ((*results["result1"])[1][0], 0) << "result1-1";
    EXPECT_EQ((*results["result1"])[2][0], 3) << "result1-2";

    EXPECT_EQ((*results["result2"])[0][0], 4) << "result2-0";
    EXPECT_EQ((*results["result2"])[1][0], 0) << "result2-1";
    EXPECT_EQ((*results["result2"])[2][0], 9) << "result2-2";
}

class DaftBatchMatrixComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    map<string, vector<vector<float>>*> results;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add two 4x4 matrices and one 4x1 column vector
        f->addOp(Operation::matrix("A1", 4, 4));
        f->addOp(Operation::matrix("A2", 4, 4));
        f->addOp(Operation::column("x", 4));
        
        // Perform sequential matrix multiplications: A1*x, then A2*(A1*x)
        f->addOp(Operation::matrixProduct("result1", "A1", "x", 4, 4, 1));
        f->addOp(Operation::matrixProduct("result2", "A2", "result1", 4, 4, 1));
        f->compile();

        // Set fixed matrix values
        f->setValue("A1", {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}});  // Identity matrix
        f->setValue("A2", {{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}});  // Cyclic permutation matrix

        results["result1"] = new vector<vector<float>>;
        results["result2"] = new vector<vector<float>>;

        // Create batch inputs with 3 different column vectors only
        map<string, vector<vector<float>>> inputs {
            {"x", {
                {1, 2, 3, 4},  // Simple sequence
                {1, 1, 1, 1},  // All ones
                {2, 3, 5, 7}   // Prime-like sequence
            }}
        };

        f->batchCompute(results, {"result1", "result2"}, inputs);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete results["result1"];
        delete results["result2"];
        delete f;
    }
};

TEST_F(DaftBatchMatrixComputeTest, BatchMatrixComputeTest) {
    // Test batch 0: Identity * [1,2,3,4] = [1,2,3,4]
    EXPECT_EQ((*results["result1"])[0][0], 1) << "A1*x batch 0, element 0";
    EXPECT_EQ((*results["result1"])[0][1], 2) << "A1*x batch 0, element 1";
    EXPECT_EQ((*results["result1"])[0][2], 3) << "A1*x batch 0, element 2";
    EXPECT_EQ((*results["result1"])[0][3], 4) << "A1*x batch 0, element 3";

    // Test batch 0: Permutation * [1,2,3,4] = [2,3,4,1]
    EXPECT_EQ((*results["result2"])[0][0], 2) << "A2*(A1*x) batch 0, element 0";
    EXPECT_EQ((*results["result2"])[0][1], 3) << "A2*(A1*x) batch 0, element 1";
    EXPECT_EQ((*results["result2"])[0][2], 4) << "A2*(A1*x) batch 0, element 2";
    EXPECT_EQ((*results["result2"])[0][3], 1) << "A2*(A1*x) batch 0, element 3";

    // Test batch 1: Identity * [1,1,1,1] = [1,1,1,1]
    EXPECT_EQ((*results["result1"])[1][0], 1) << "A1*x batch 1, element 0";
    EXPECT_EQ((*results["result1"])[1][1], 1) << "A1*x batch 1, element 1";
    EXPECT_EQ((*results["result1"])[1][2], 1) << "A1*x batch 1, element 2";
    EXPECT_EQ((*results["result1"])[1][3], 1) << "A1*x batch 1, element 3";

    // Test batch 1: Permutation * [1,1,1,1] = [1,1,1,1]
    EXPECT_EQ((*results["result2"])[1][0], 1) << "A2*(A1*x) batch 1, element 0";
    EXPECT_EQ((*results["result2"])[1][1], 1) << "A2*(A1*x) batch 1, element 1";
    EXPECT_EQ((*results["result2"])[1][2], 1) << "A2*(A1*x) batch 1, element 2";
    EXPECT_EQ((*results["result2"])[1][3], 1) << "A2*(A1*x) batch 1, element 3";

    // Test batch 2: Identity * [2,3,5,7] = [2,3,5,7]
    EXPECT_EQ((*results["result1"])[2][0], 2) << "A1*x batch 2, element 0";
    EXPECT_EQ((*results["result1"])[2][1], 3) << "A1*x batch 2, element 1";
    EXPECT_EQ((*results["result1"])[2][2], 5) << "A1*x batch 2, element 2";
    EXPECT_EQ((*results["result1"])[2][3], 7) << "A1*x batch 2, element 3";

    // Test batch 2: Permutation * [2,3,5,7] = [3,5,7,2]
    EXPECT_EQ((*results["result2"])[2][0], 3) << "A2*(A1*x) batch 2, element 0";
    EXPECT_EQ((*results["result2"])[2][1], 5) << "A2*(A1*x) batch 2, element 1";
    EXPECT_EQ((*results["result2"])[2][2], 7) << "A2*(A1*x) batch 2, element 2";
    EXPECT_EQ((*results["result2"])[2][3], 2) << "A2*(A1*x) batch 2, element 3";
}
