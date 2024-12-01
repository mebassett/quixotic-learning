#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftScalarTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function* f;
    float* result;
    void SetUp() override
    {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        f->addOp(Operation::column("xy",2));
        f->addOp(Operation::scalarMultiply("result","xy",2,1,5.0));
        f->compile();
        result = new float[2];

    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete [] result;
        delete f;
    }
};

TEST_F(DaftScalarTest, DaftScalarCompute){
    f->setValue("xy", {1,2});
    f->compute();
    f->getValue("result", result);
    EXPECT_EQ(result[0],5);
    EXPECT_EQ(result[1],10);
}


class DaftInnerProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function* f;
    Function* g;
    float* result;
    float* result2;
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
        result = new float[1];
        result2 = new float;
    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete [] result;
        delete result2;
        delete f;
        delete g;

    }
};

TEST_F(DaftInnerProductTest, DaftInnerProductCompute) {
    f->setValue("ab", {3.0, 4.0});
    f->setValue("xy", {1.0, 2.0});
    f->compute();
    f->getValue("test1", result);
    EXPECT_EQ(result[0], 11.0) << "compute";

    g->setValue("x", {9});
    g->compute();
    g->computeGrad("test1");
    g->getGrad("x", result2);
    EXPECT_EQ(result2[0], 18) << "x grad";
}
