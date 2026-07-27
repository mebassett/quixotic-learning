#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftMatrixColProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;
    Function *g;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::weightsMatrix("abcd", 2, 2));
        f->addOp(Operation::matrixProduct("f", "abcd", "xy", 2, 2, 1));
        f->compile(); 

        g = new Function(&cublasH);
        g->addOp(Operation::weightsMatrix("A", 2, 2));
        g->addOp(Operation::inputMatrix("B", 2, 2));
        g->addOp(Operation::matrixProduct("g", "A", "B", 2, 2, 2));
        g-> compile();

    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
        delete g;
    }
};

TEST_F(DaftMatrixColProductTest, DaftMatrixColProductCompute) {
    f->setValue("abcd", {{1,-1,-1,1}});
    f->setValue("xy", {{1,2}});
    f->compute();
    f->computeGrad("f");
    vector<vector<float>> resultVec;
    vector<vector<float>> matrixGrad;
    f->getValue("f", &resultVec);
    f->getGrad("abcd", &matrixGrad);
    EXPECT_EQ(resultVec[0][0],-1) << "compute0";
    EXPECT_EQ(resultVec[0][1],1) << "compute1";
    EXPECT_EQ(matrixGrad[0][0], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[0][1], 2) << "abcd grad";
    EXPECT_EQ(matrixGrad[0][2], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[0][3], 2) << "abcd grad";

    g->setValue("A", {{1,2,3,4}});
    g->setValue("B", {{1,1,-1,1}});
    g->compute();
    vector<vector<float>> resultVec2;
    g->getValue("g", &resultVec2);
    EXPECT_EQ(resultVec2[0][0],-1) << "AB00";
    EXPECT_EQ(resultVec2[0][1],3) << "AB01";
    EXPECT_EQ(resultVec2[0][2],-1) << "AB10";
    EXPECT_EQ(resultVec2[0][3],7) << "AB11";
}

class DaftBatchMatrixColProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;
    Function *g;

    void SetUp() override {
        cublasCreate(&cublasH);

        // f: fixed 2x2 matrix * batch of 2x1 columns -> 2x1 result
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::weightsMatrix("abcd", 2, 2));
        f->addOp(Operation::matrixProduct("f", "abcd", "xy", 2, 2, 1));
        f->compile(3);  // Compile for batch size of 3

        // g: fixed 2x2 matrix * batch of 2x2 matrices -> 2x2 result
        g = new Function(&cublasH);
        g->addOp(Operation::weightsMatrix("A", 2, 2));
        g->addOp(Operation::inputMatrix("B", 2, 2));
        g->addOp(Operation::matrixProduct("g", "A", "B", 2, 2, 2));
        g->compile(2);  // Compile for batch size of 2
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
        delete g;
    }
};

TEST_F(DaftBatchMatrixColProductTest, BatchMatrixColProductCompute) {
    // f test: abcd = [[1,-1],[-1,1]] (fixed InputMatrix)
    f->setValue("abcd", {{1, -1, -1, 1}});
    f->setValue("xy", {
        {1, 2},     // batch 0
        {3, 4},     // batch 1
        {-1, 2}     // batch 2
    });
    f->compute();
    vector<vector<float>> resultVec;
    f->getValue("f", &resultVec);

    // batch 0: [[1,-1],[-1,1]] * [1,2] = [1*1+(-1)*2, (-1)*1+1*2] = [-1, 1]
    // batch 1: [[1,-1],[-1,1]] * [3,4] = [1*3+(-1)*4, (-1)*3+1*4] = [-1, 1]
    // batch 2: [[1,-1],[-1,1]] * [-1,2] = [(-1)+(-2), 1+2] = [-3, 3]
    float expected[3][2] = {
        {-1, 1},
        {-1, 1},
        {-3, 3}
    };
    for (int batch = 0; batch < 3; batch++) {
        for (int i = 0; i < 2; i++) {
            EXPECT_EQ(resultVec[batch][i], expected[batch][i])
                << "Batch matrix-col product batch " << batch << ", element " << i;
        }
    }

    // g test: A = [[1,2],[3,4]] (fixed InputMatrix)
    g->setValue("A", {{1, 2, 3, 4}});
    g->setValue("B", {
        {1, 1, -1, 1},   // batch 0
        {2, 0, 1, -1}    // batch 1
    });
    g->compute();
    vector<vector<float>> resultVec2;
    g->getValue("g", &resultVec2);

    // batch 0: A * [[1,1],[-1,1]] = [[1*1+2*(-1), 1*1+2*1], [3*1+4*(-1), 3*1+4*1]] = [[-1,3],[-1,7]]
    // batch 1: A * [[2,0],[1,-1]] = [[1*2+2*1, 1*0+2*(-1)], [3*2+4*1, 3*0+4*(-1)]] = [[4,-2],[10,-4]]
    float expected2[2][4] = {
        {-1, 3, -1, 7},
        {4, -2, 10, -4}
    };
    for (int batch = 0; batch < 2; batch++) {
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(resultVec2[batch][i], expected2[batch][i])
                << "Batch matrix-matrix product batch " << batch << ", element " << i;
        }
    }
}
