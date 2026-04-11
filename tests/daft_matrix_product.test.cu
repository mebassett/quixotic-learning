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
        f->addOp(Operation::matrix("abcd", 2, 2));
        f->addOp(Operation::matrixProduct("f", "abcd", "xy", 2, 2, 1));
        f->compile(); 

        g = new Function(&cublasH);
        g->addOp(Operation::matrix("A", 2, 2));
        g->addOp(Operation::matrix("B", 2, 2));
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
