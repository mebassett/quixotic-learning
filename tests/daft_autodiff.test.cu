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
    float* result;
    float* result2;
    float* result3;
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
        result = new float[1];
        result2 = new float[1];
        result3 = new float[2];
    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete [] result;
        delete [] result2;
        delete [] result3;
        delete f;
        delete g;
        delete h;

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
    EXPECT_EQ(result2[0], 18) << "x0 grad";

    h->setValue("sr", {1.0,2.0});
    h->setValue("tu", {3.0,-3.0});
    h->compute();
    h->getValue("test2", result2);
    EXPECT_EQ(*result2, -3.0) << "compute";

    h->computeGrad("test2");
    h->getGrad("sr", result3);
    EXPECT_EQ(result3[0], 3) << "s grad ";
    EXPECT_EQ(result3[1], -3) << "r grad ";
}

class DaftMatrixColProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;
    float *result;
    float *matrixGrad;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::matrix("abcd", 2, 2));
        f->addOp(Operation::matrixProduct("f", "abcd", "xy", 2, 2, 1));
        f->compile(); 

        result = new float[2];
        matrixGrad = new float[4];
    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete [] result;
        delete [] matrixGrad;
        delete f;
    }
};

TEST_F(DaftMatrixColProductTest, DaftMatrixColProductCompute) {
    f->setValue("abcd", {1,-1,-1,1});
    f->setValue("xy", {1,2});
    f->compute();
    f->computeGrad("f");
    f->getValue("f", result);
    f->getGrad("abcd", matrixGrad);
    EXPECT_EQ(result[0],-1) << "compute0";
    EXPECT_EQ(result[1],1) << "compute1";
    EXPECT_EQ(matrixGrad[0], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[1], 2) << "abcd grad";
    EXPECT_EQ(matrixGrad[2], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[3], 2) << "abcd grad";


}

class DaftScalarTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function *f;
    float *result;
    float *resultGrad;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::scalarMultiply("test", "xy", 2, 1, 5.0));
        f->compile();

        result = new float[2];
        resultGrad = new float[2];
    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete [] result;
        delete [] resultGrad;
        delete f;
    }
};

TEST_F(DaftScalarTest, DaftScalarCompute) {
    f->setValue("xy",{1,2});
    f->compute();
    f->computeGrad("test");
    f->getValue("test", result);
    f->getGrad("xy", resultGrad);
    EXPECT_EQ(result[0], 5) << "compute0";
    EXPECT_EQ(result[1], 10) << "compute1";
    EXPECT_EQ(resultGrad[0], 5) << "grad0";
    EXPECT_EQ(resultGrad[1], 5) << "grad1";
}

class DaftAddTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function *f;
    float *result;
    float *resultGrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy",2));
        f->addOp(Operation::add("f","xy","xy", 2, 1));
        f->compile();

        result = new float[2];
        resultGrad = new float[2];
    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete [] result;
        delete [] resultGrad;
        delete f;
    }
};
TEST_F(DaftAddTest, DaftAddCompute) {
    f->setValue("xy",{1,2});
    f->compute();
    f->computeGrad("f");
    f->getValue("f", result);
    f->getGrad("xy", resultGrad);
    EXPECT_EQ(result[0], 2) << "compute0";
    EXPECT_EQ(result[1], 4) << "compute1";
    EXPECT_EQ(resultGrad[0], 2) << "grad0";
    EXPECT_EQ(resultGrad[1], 2) << "grad1";
}

class DaftLeakyReLUTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function *f;
    float *result;
    float *resultGrad;

    void SetUp() override {
        cublasCreate(&cublasH);

        f = new Function(&cublasH);
        f->addOp(Operation::matrix("z",2,2));
        f->addOp(Operation::applyLeakyReLU("f", "z", 2,2));
        f->compile();

        result = new float[4];
        resultGrad = new float[4];
    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete [] result;
        delete [] resultGrad;
        delete f;
    }
};

TEST_F(DaftLeakyReLUTest, DaftLeakyReLUCompute) {
    f->setValue("z", { 500, -500, 0.5, -1 });
    f->compute();
    f->computeGrad("f");
    f->getValue("f", result);
    f->getGrad("z", resultGrad);

    float values[4] = { 500, -5, 0.5, -0.01 };
    float grads[4] = { 1, 0.01, 1, 0.01 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(result[i], values[i]) << "LeakyReLU compute (" << i << ")";
        EXPECT_EQ(resultGrad[i], grads[i]) << "z grad (" << i << ")";
    }

}
