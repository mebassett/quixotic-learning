#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;
class DaftInnerProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function* f;
    Function* g;
    Function* h;
    vector<vector<float>> result;
    vector<vector<float>> result2;
    vector<vector<float>> result3;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("ab", 2));
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::innerProduct("test1", "ab", "xy", 2));
        f->compile();
        g = new Function(&cublasH);
        g->addOp(Operation::column("x", 1));
        g->addOp(Operation::innerProduct("test1", "x", "x", 1));
        g->compile();
        h = new Function(&cublasH);
        h->addOp(Operation::column("sr", 2));
        h->addOp(Operation::column("tu", 2));
        h->addOp(Operation::innerProduct("test2", "sr", "tu", 2));
        h->compile();
    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
        delete g;
        delete h;

    }
};

TEST_F(DaftInnerProductTest, DaftInnerProductCompute) {
    f->setValue("ab", {{3.0, 4.0}});
    f->setValue("xy", {{1.0, 2.0}});
    f->compute();
    vector<vector<float>> resultVec;
    f->getValue("test1", &resultVec);
    EXPECT_EQ(resultVec[0][0], 11.0) << "compute";

    g->setValue("x", {{9}});
    g->compute();
    g->computeGrad("test1");
    g->getGrad("x", &result2);
    EXPECT_EQ(result2[0][0], 18) << "x0 grad";

    h->setValue("sr", {{1.0,2.0}});
    h->setValue("tu", {{3.0,-3.0}});
    h->compute();
    vector<vector<float>> resultVec2;
    h->getValue("test2", &resultVec2);
    EXPECT_EQ(resultVec2[0][0], -3.0) << "compute";

    h->computeGrad("test2");
    h->getGrad("sr", &result3);
    EXPECT_EQ(result3[0][0], 3) << "s grad ";
    EXPECT_EQ(result3[0][1], -3) << "r grad ";
}
