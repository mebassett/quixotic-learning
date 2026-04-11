#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftConvolutionTestNoPaddingSingleOffset : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> result;

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

        // Set up values
        f->setValue("input", {{1, 2, 3, 4, 5, 6, 7, 8, 9}});
        f->setValue("kernel", {{3, 3, 3, 3}});

        f->compute();
        
        f->getValue("conv", &result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftConvolutionTestNoPaddingSingleOffset, ConvolutionTestCompute) {
    float values[4] = {36, 48, 72, 84};
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(result[0][i], values[i])
            << "Daft Convolution compute, no padding single offset.";
}

class DaftConvolutionTestPaddedWithStride : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> result;

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

        // Set up values
        f->setValue("input", {{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7}});
        f->setValue("kernel", {{1, 0, 0, 0, 1, 0, 0, 0, 1}}); 

        f->compute();
        
        f->getValue("conv", &result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftConvolutionTestPaddedWithStride, ConvolutionTestCompute) {
    float values[4] = {7, 4, 4, 9};
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(result[0][i], values[i])
            << "Daft Convolution compute, 1 padding, 3 stride.";
}

class DaftConvolutionGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> kernelGrad;
    vector<vector<float>> inputGrad;

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

        // Set up values
        f->setValue("input", {{1, 2, 3, 4}});
        f->setValue("kernel", {{3, 3, 3, 3}});

        f->compute();
        f->computeGrad("conv");
        
        f->getGrad("kernel", &kernelGrad);
        f->getGrad("input", &inputGrad);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftConvolutionGradTest, ConvolutionGradTestCompute) {
    float kernelGradValues[4] = {1, 2, 3, 4};
    float inputGradValues[4] = {3, 3, 3, 3};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[0][i], kernelGradValues[i]) << "Daft Convolution kernel grad";
        EXPECT_EQ(inputGrad[0][i], inputGradValues[i]) << "Daft Convolution input grad";
    }
}

class DaftConvolutionDoubleGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> kernelGrad;
    vector<vector<float>> result;

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

        // Set up values
        f->setValue("id3", {{0, 1, -1, 0}});
        f->setValue("k2", {{5, 6, 9, 3}});
        f->setValue("k3", {{1, 1, 1, 1}});

        f->compute();
        f->computeGrad("f2");
        
        f->getGrad("k2", &kernelGrad);
        f->getValue("f2", &result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftConvolutionDoubleGradTest, ConvolutionDoubleGradTestCompute) {
    float kernelGradValues[4] = {0, -1, 1, 0};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[0][i], kernelGradValues[i])
            << "Daft Convolution*Convolution kernel grad";
    }
}

class DaftConvolutionGradInnerProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> kernelGrad;
    vector<vector<float>> result;

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

        // Set up values
        f->setValue("id3", {{1, 0, 0, 0, 1, 0, 0, 0, 1}});  // 3x3 identity
        f->setValue("k2", {{0, 1, 1, 0}});                   // 2x2 kernel
        f->setValue("v", {{1, 1}});                          // 2-element vector

        f->compute();
        f->computeGrad("f1");
        
        f->getGrad("k2", &kernelGrad);
        f->getValue("f1", &result);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftConvolutionGradInnerProductTest, ConvolutionGradInnerProductTestCompute) {
    EXPECT_EQ(result[0][0], 2) << "Daft Convolution*InnerProduct value";
    float kernelGradValues[4] = {4, 2, 2, 4};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[0][i], kernelGradValues[i])
            << "Daft Convolution*InnerProduct kernel grad";
    }
}
