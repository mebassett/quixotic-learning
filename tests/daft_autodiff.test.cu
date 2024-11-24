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
    f->getValue(result);
    EXPECT_EQ(result[0],5);
    EXPECT_EQ(result[1],10);
}

