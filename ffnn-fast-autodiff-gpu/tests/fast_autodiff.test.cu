#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <valarray>
#include "../fast_autodiff/fast_autodiff.h"

using namespace FA;

class ScalarTest : public testing::Test {
    protected:
        cublasHandle_t cublasH;

        Col* xy;
        Scalar* test_scalar;
        void SetUp() override {
            cublasCreate(&cublasH);

            this->xy = new Col("xy",2);
            this->test_scalar = new Scalar(xy, 5);

        }
        void TearDown() override {
            cublasDestroy(cublasH);
            delete this->test_scalar;
        }

};

TEST_F(ScalarTest, ScalarCompute) {
    xy->loadValues({1,2});
    test_scalar->compute(&cublasH);
    test_scalar->fromDevice();
    EXPECT_EQ(test_scalar->value[0], 5);
    EXPECT_EQ(test_scalar->value[1], 10);
}

class InnerProductTest : public testing::Test {
    protected:
        cublasHandle_t cublasH;

        Col* ab ;
        Col* xy;
        InnerProduct* test_ip ;
        void SetUp() override {
            cublasCreate(&cublasH);
            ab = new Col("ab", 2);
            this->xy = new Col("xy",2);
            this->test_ip = new InnerProduct(xy, ab);

        }
        void TearDown() override {
            cublasDestroy(cublasH);
            delete this->test_ip;
        }
            
};

TEST_F(InnerProductTest, InnerProductCompute){
    ab->loadValues({ 3.0, 4.0 });
    xy->loadValues({ 1.0, 2.0});
    test_ip->compute(&cublasH);
    test_ip->fromDevice();
    EXPECT_EQ(test_ip->value[0], 11.0);
}
