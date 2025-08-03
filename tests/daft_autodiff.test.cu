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
    Function *g;
    float *result;
    float *matrixGrad;
    float *result2;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::matrix("abcd", 2, 2));
        f->addOp(Operation::matrixProduct("f", "abcd", "xy", 2, 2, 1));
        f->compile(); 

        result = new float[2];
        matrixGrad = new float[4];

        g = new Function(&cublasH);
        g->addOp(Operation::matrix("A", 2, 2));
        g->addOp(Operation::matrix("B", 2, 2));
        g->addOp(Operation::matrixProduct("g", "A", "B", 2, 2, 2));
        g-> compile();
        result2 = new float[4];

    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete [] result;
        delete [] result2;
        delete [] matrixGrad;
        delete f;
        delete g;
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

    g->setValue("A", {1,2,3,4});
    g->setValue("B", {1,1,-1,1});
    g->compute();
    g->getValue("g", result2);
    EXPECT_EQ(result2[0],-1) << "AB00";
    EXPECT_EQ(result2[1],3) << "AB01";
    EXPECT_EQ(result2[2],-1) << "AB10";
    EXPECT_EQ(result2[3],7) << "AB11";



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

class DaftConvolutionTestNoPaddingSingleOffset : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* result;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add input matrix (3x3) and kernel (2x2)
        f->addOp(Operation::matrix("input", 3, 3));
        f->addOp(Operation::matrix("kernel", 2, 2));
        
        // Add convolution operation with no padding (0,0) and single offset (1,1)
        f->addOp(Operation::convolution("conv", "input", "kernel", 
                                      0, 1, 0, 1,  // rowPadding, rowSkip, colPadding, colSkip
                                      3, 3,        // multiplicandRows, multiplicandCols
                                      2, 2));      // kernelRows, kernelCols
        f->compile();

        // Set up values - same as silly test
        f->setValue("input", {1, 2, 3, 4, 5, 6, 7, 8, 9});
        f->setValue("kernel", {3, 3, 3, 3});

        f->compute();
        
        result = new float[4];  // 2x2 output
        f->getValue("conv", result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] result;
        delete f;
    }
};

TEST_F(DaftConvolutionTestNoPaddingSingleOffset, ConvolutionTestCompute) {
    float values[4] = {36, 48, 72, 84};
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(result[i], values[i])
            << "Daft Convolution compute, no padding single offset.";
}

class DaftConvolutionTestPaddedWithStride : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* result;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add input matrix (4x4) and kernel (3x3)
        f->addOp(Operation::matrix("input", 4, 4));
        f->addOp(Operation::matrix("kernel", 3, 3));
        
        // Add convolution operation with padding (1,1) and stride (3,3)
        f->addOp(Operation::convolution("conv", "input", "kernel", 
                                      1, 3, 1, 3,  // rowPadding, rowSkip, colPadding, colSkip
                                      4, 4,        // multiplicandRows, multiplicandCols
                                      3, 3));      // kernelRows, kernelCols
        f->compile();

        // Set up values - same as silly test
        f->setValue("input", {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7});
        f->setValue("kernel", {1, 0, 0, 0, 1, 0, 0, 0, 1}); 

        f->compute();
        
        result = new float[4];  // 2x2 output
        f->getValue("conv", result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] result;
        delete f;
    }
};

TEST_F(DaftConvolutionTestPaddedWithStride, ConvolutionTestCompute) {
    float values[4] = {7, 4, 4, 9};
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(result[i], values[i])
            << "Daft Convolution compute, 1 padding, 3 stride.";
}

class DaftConvolutionGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* kernelGrad;
    float* inputGrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add input matrix (2x2) and kernel (2x2)
        f->addOp(Operation::matrix("input", 2, 2));
        f->addOp(Operation::matrix("kernel", 2, 2));
        
        // Add convolution operation with no padding (0,0) and single offset (1,1)
        f->addOp(Operation::convolution("conv", "input", "kernel", 
                                      0, 1, 0, 1,  // rowPadding, rowSkip, colPadding, colSkip
                                      2, 2,        // multiplicandRows, multiplicandCols
                                      2, 2));      // kernelRows, kernelCols
        f->compile();

        // Set up values - same as silly test
        f->setValue("input", {1, 2, 3, 4});
        f->setValue("kernel", {3, 3, 3, 3});

        f->compute();
        f->computeGrad("conv");
        
        kernelGrad = new float[4];
        inputGrad = new float[4];
        f->getGrad("kernel", kernelGrad);
        f->getGrad("input", inputGrad);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] kernelGrad;
        delete[] inputGrad;
        delete f;
    }
};

TEST_F(DaftConvolutionGradTest, ConvolutionGradTestCompute) {
    float kernelGradValues[4] = {1, 2, 3, 4};
    float inputGradValues[4] = {3, 3, 3, 3};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[i], kernelGradValues[i]) << "Daft Convolution kernel grad";
        EXPECT_EQ(inputGrad[i], inputGradValues[i]) << "Daft Convolution input grad";
    }
}

class DaftConvolutionGradInnerProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* kernelGrad;
    float* result;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add 3x3 identity matrix, 2x2 kernel, and 2-element vector
        f->addOp(Operation::matrix("id3", 3, 3));
        f->addOp(Operation::matrix("k2", 2, 2));
        f->addOp(Operation::column("v", 2));
        
        // Add convolution (3x3 -> 2x2 output)
        f->addOp(Operation::convolution("c2", "id3", "k2", 
                                      0, 1, 0, 1,  // no padding, stride 1
                                      3, 3,        // input 3x3
                                      2, 2));      // kernel 2x2
        
        // Add matrix-column product (2x2 output becomes 2x1)
        f->addOp(Operation::matrixProduct("p", "c2", "v", 2, 2, 1));
        
        // Add inner product (2x1 with itself -> scalar)
        f->addOp(Operation::innerProduct("f1", "p", "p", 2));
        
        f->compile();

        // Set up values - same as silly test
        f->setValue("id3", {1, 0, 0, 0, 1, 0, 0, 0, 1});  // 3x3 identity
        f->setValue("k2", {0, 1, 1, 0});                   // 2x2 kernel
        f->setValue("v", {1, 1});                          // 2-element vector

        f->compute();
        f->computeGrad("f1");
        
        kernelGrad = new float[4];
        result = new float[1];
        f->getGrad("k2", kernelGrad);
        f->getValue("f1", result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] kernelGrad;
        delete[] result;
        delete f;
    }
};

TEST_F(DaftConvolutionGradInnerProductTest, ConvolutionGradInnerProductTestCompute) {
    EXPECT_EQ(result[0], 2) << "Daft Convolution*InnerProduct value";
    float kernelGradValues[4] = {4, 2, 2, 4};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[i], kernelGradValues[i])
            << "Daft Convolution*InnerProduct kernel grad";
    }
}

class DaftConvolutionDoubleGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* kernelGrad;
    float* result;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add 2x2 input matrix and two 2x2 kernels
        f->addOp(Operation::matrix("id3", 2, 2));
        f->addOp(Operation::matrix("k2", 2, 2));
        f->addOp(Operation::matrix("k3", 2, 2));
        
        // First convolution with padding and stride 2
        f->addOp(Operation::convolution("c2", "id3", "k2", 
                                      1, 2, 1, 2,  // padding 1, stride 2
                                      2, 2,        // input 2x2
                                      2, 2));      // kernel 2x2
        
        // Second convolution with no padding, stride 1
        f->addOp(Operation::convolution("f2", "c2", "k3", 
                                      0, 1, 0, 1,  // no padding, stride 1
                                      2, 2,        // c2 output is 2x2
                                      2, 2));      // kernel 2x2
        
        f->compile();

        // Set up values - same as silly test
        f->setValue("id3", {0, 1, -1, 0});
        f->setValue("k2", {5, 6, 9, 3});
        f->setValue("k3", {1, 1, 1, 1});

        f->compute();
        f->computeGrad("f2");
        
        kernelGrad = new float[4];
        result = new float[1];
        f->getGrad("k2", kernelGrad);
        f->getValue("f2", result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] kernelGrad;
        delete[] result;
        delete f;
    }
};

TEST_F(DaftConvolutionDoubleGradTest, ConvolutionDoubleGradTestCompute) {
    float kernelGradValues[4] = {0, -1, 1, 0};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[i], kernelGradValues[i])
            << "Daft Convolution*Convolution kernel grad";
    }
}

class DaftMaxPoolComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* result;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add 2x2 input matrix
        f->addOp(Operation::matrix("id3", 2, 2));
        
        // Add MaxPool operation with 2x2 pool size and stride 1
        f->addOp(Operation::maxPool("mp", "id3", 2, 2, 1, 1, 2, 2));
        
        f->compile();

        // Set up values - same as silly test
        f->setValue("id3", {1, 2, 3, 4});

        f->compute();
        
        result = new float[1];  // 1x1 output
        f->getValue("mp", result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] result;
        delete f;
    }
};

TEST_F(DaftMaxPoolComputeTest, MaxPoolComputeTest) {
    EXPECT_EQ(result[0], 4) << "Daft MaxPool compute";
}

class DaftMaxPoolLargeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* result;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add 4x4 input matrix
        f->addOp(Operation::matrix("id3", 4, 4));
        
        // Add MaxPool operation with 2x2 pool size and stride 2
        f->addOp(Operation::maxPool("mp", "id3", 2, 2, 2, 2, 4, 4));
        
        f->compile();

        // Set up values - same as silly test
        f->setValue("id3", {1, 2, 1, 2, 3, 9, 16, 3, 1, 10, 4, 1, 3, 4, 2, 3});

        f->compute();
        
        result = new float[4];  // 2x2 output
        f->getValue("mp", result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] result;
        delete f;
    }
};

TEST_F(DaftMaxPoolLargeTest, MaxPoolLargeTest) {
    float values[4] = {9, 16, 10, 4};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(result[i], values[i]) << "Daft MaxPool large test";
    }
}

class DaftMaxPoolGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* testvalue;
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
        
        f->compile();

        // Set up values - same as silly test
        f->setValue("id3", {1, 1, 1, 4});

        f->compute();
        f->computeGrad("smp");
        
        testvalue = new float[4];
        f->getGrad("id3", testvalue);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] testvalue;
        delete f;
    }
};

TEST_F(DaftMaxPoolGradTest, MaxPoolGradTest) {
    float values[4] = {0, 0, 0, scalarValue};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(testvalue[i], values[i]) << "Daft MaxPool grad test";
    }
}

class DaftConcatComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    float* result;
    float* testgrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add two single-element columns and one 2-element column
        f->addOp(Operation::column("v2", 1));
        f->addOp(Operation::column("v3", 1));
        f->addOp(Operation::column("v4", 2));
        
        // Add concat operation to combine v2 and v3 into a 2-element vector
        f->addOp(Operation::concat("v2v3", {"v2", "v3"}, 2));
        
        // Add inner product between the concatenated vector and v4
        f->addOp(Operation::innerProduct("flatF", "v2v3", "v4", 2));
        
        f->compile();

        // Set up values - same as silly test
        f->setValue("v2", {1});
        f->setValue("v3", {2});
        f->setValue("v4", {3, 5});

        f->compute();
        f->computeGrad("flatF");
        
        result = new float[1];
        testgrad = new float[1];
        f->getValue("flatF", result);
        f->getGrad("v2", testgrad);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete[] result;
        delete[] testgrad;
        delete f;
    }
};

TEST_F(DaftConcatComputeTest, ConcatComputeTest) {
    EXPECT_EQ(result[0], 13) << "Daft Concat compute";
    EXPECT_EQ(testgrad[0], 3) << "Daft Concat grad";
}
