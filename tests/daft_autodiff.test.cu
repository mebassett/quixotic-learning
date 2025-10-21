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

class DaftMatrixColProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;
    Function *g;
    vector<vector<float>> result;
    vector<vector<float>> matrixGrad;
    vector<vector<float>> result2;
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
    //f->computeGrad("f");
    vector<vector<float>> resultVec;
    f->getValue("f", &resultVec);
    //f->getGrad("abcd", matrixGrad);
    EXPECT_EQ(resultVec[0][0],-1) << "compute0";
    EXPECT_EQ(resultVec[0][1],1) << "compute1";
    //EXPECT_EQ(matrixGrad[0], 1) << "abcd grad";
    //EXPECT_EQ(matrixGrad[1], 2) << "abcd grad";
    //EXPECT_EQ(matrixGrad[2], 1) << "abcd grad";
    //EXPECT_EQ(matrixGrad[3], 2) << "abcd grad";

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

class DaftScalarTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function *f;
    vector<vector<float>> result;
    vector<vector<float>> resultGrad;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::scalarMultiply("test", "xy", 2, 1, 5.0));
        f->compile();

    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftScalarTest, DaftScalarCompute) {
    f->setValue("xy",{{1,2}});
    f->compute();
    f->computeGrad("test");
    vector<vector<float>> resultVec;
    f->getValue("test", &resultVec);
    f->getGrad("xy", &resultGrad);
    EXPECT_EQ(resultVec[0][0], 5) << "compute0";
    EXPECT_EQ(resultVec[0][1], 10) << "compute1";
    EXPECT_EQ(resultGrad[0][0], 5) << "grad0";
    EXPECT_EQ(resultGrad[0][1], 5) << "grad1";
}

class DaftAddTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function *f;
    vector<vector<float>> result;
    vector<vector<float>> resultGrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy",2));
        f->addOp(Operation::add("f","xy","xy", 2, 1));
        f->compile();

    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};
TEST_F(DaftAddTest, DaftAddCompute) {
    f->setValue("xy",{{1,2}});
    f->compute();
    f->computeGrad("f");
    vector<vector<float>> resultVec;
    f->getValue("f", &resultVec);
    f->getGrad("xy", &resultGrad);
    EXPECT_EQ(resultVec[0][0], 2) << "compute0";
    EXPECT_EQ(resultVec[0][1], 4) << "compute1";
    EXPECT_EQ(resultGrad[0][0], 2) << "grad0";
    EXPECT_EQ(resultGrad[0][1], 2) << "grad1";
}

class DaftLeakyReLUTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Function *f;
    vector<vector<float>> result;
    vector<vector<float>> resultGrad;

    void SetUp() override {
        cublasCreate(&cublasH);

        f = new Function(&cublasH);
        f->addOp(Operation::matrix("z",2,2));
        f->addOp(Operation::applyLeakyReLU("f", "z", 2,2));
        f->compile();

    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftLeakyReLUTest, DaftLeakyReLUCompute) {
    f->setValue("z", {{ 500, -500, 0.5, -1 }});
    f->compute();
    f->computeGrad("f");
    vector<vector<float>> resultVec;
    f->getValue("f", &resultVec);
    f->getGrad("z", &resultGrad);

    float values[4] = { 500, -5, 0.5, -0.01 };
    float grads[4] = { 1, 0.01, 1, 0.01 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(resultVec[0][i], values[i]) << "LeakyReLU compute (" << i << ")";
        EXPECT_EQ(resultGrad[0][i], grads[i]) << "z grad (" << i << ")";
    }

}

// class DaftConvolutionTestNoPaddingSingleOffset : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     float* result;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add input matrix (3x3) and kernel (2x2)
//         f->addOp(Operation::matrix("input", 3, 3));
//         f->addOp(Operation::matrix("kernel", 2, 2));
//         
//         // Add convolution operation with no padding (0,0) and single offset (1,1)
//         f->addOp(Operation::convolution("conv", "input", "kernel", 
//                                       0, 1, 0, 1,  // rowPadding, rowSkip, colPadding, colSkip
//                                       3, 3,        // multiplicandRows, multiplicandCols
//                                       2, 2));      // kernelRows, kernelCols
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("input", {1, 2, 3, 4, 5, 6, 7, 8, 9});
//         f->setValue("kernel", {3, 3, 3, 3});
// 
//         f->compute();
//         
//         result = new float[4];  // 2x2 output
//         f->getValue("conv", result);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete[] result;
//         delete f;
//     }
// };
// 
// TEST_F(DaftConvolutionTestNoPaddingSingleOffset, ConvolutionTestCompute) {
//     float values[4] = {36, 48, 72, 84};
//     for (int i = 0; i < 4; i++)
//         EXPECT_EQ(result[i], values[i])
//             << "Daft Convolution compute, no padding single offset.";
// }
// 
// class DaftConvolutionTestPaddedWithStride : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     float* result;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add input matrix (4x4) and kernel (3x3)
//         f->addOp(Operation::matrix("input", 4, 4));
//         f->addOp(Operation::matrix("kernel", 3, 3));
//         
//         // Add convolution operation with padding (1,1) and stride (3,3)
//         f->addOp(Operation::convolution("conv", "input", "kernel", 
//                                       1, 3, 1, 3,  // rowPadding, rowSkip, colPadding, colSkip
//                                       4, 4,        // multiplicandRows, multiplicandCols
//                                       3, 3));      // kernelRows, kernelCols
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("input", {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7});
//         f->setValue("kernel", {1, 0, 0, 0, 1, 0, 0, 0, 1}); 
// 
//         f->compute();
//         
//         result = new float[4];  // 2x2 output
//         f->getValue("conv", result);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete[] result;
//         delete f;
//     }
// };
// 
// TEST_F(DaftConvolutionTestPaddedWithStride, ConvolutionTestCompute) {
//     float values[4] = {7, 4, 4, 9};
//     for (int i = 0; i < 4; i++)
//         EXPECT_EQ(result[i], values[i])
//             << "Daft Convolution compute, 1 padding, 3 stride.";
// }
// 
// class DaftConvolutionGradTest : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     float* kernelGrad;
//     float* inputGrad;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add input matrix (2x2) and kernel (2x2)
//         f->addOp(Operation::matrix("input", 2, 2));
//         f->addOp(Operation::matrix("kernel", 2, 2));
//         
//         // Add convolution operation with no padding (0,0) and single offset (1,1)
//         f->addOp(Operation::convolution("conv", "input", "kernel", 
//                                       0, 1, 0, 1,  // rowPadding, rowSkip, colPadding, colSkip
//                                       2, 2,        // multiplicandRows, multiplicandCols
//                                       2, 2));      // kernelRows, kernelCols
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("input", {1, 2, 3, 4});
//         f->setValue("kernel", {3, 3, 3, 3});
// 
//         f->compute();
//         f->computeGrad("conv");
//         
//         kernelGrad = new float[4];
//         inputGrad = new float[4];
//         f->getGrad("kernel", kernelGrad);
//         f->getGrad("input", inputGrad);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete[] kernelGrad;
//         delete[] inputGrad;
//         delete f;
//     }
// };
// 
// TEST_F(DaftConvolutionGradTest, ConvolutionGradTestCompute) {
//     float kernelGradValues[4] = {1, 2, 3, 4};
//     float inputGradValues[4] = {3, 3, 3, 3};
//     for (int i = 0; i < 4; i++) {
//         EXPECT_EQ(kernelGrad[i], kernelGradValues[i]) << "Daft Convolution kernel grad";
//         EXPECT_EQ(inputGrad[i], inputGradValues[i]) << "Daft Convolution input grad";
//     }
// }
// 
// class DaftConvolutionGradInnerProductTest : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     float* kernelGrad;
//     float* result;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add 3x3 identity matrix, 2x2 kernel, and 2-element vector
//         f->addOp(Operation::matrix("id3", 3, 3));
//         f->addOp(Operation::matrix("k2", 2, 2));
//         f->addOp(Operation::column("v", 2));
//         
//         // Add convolution (3x3 -> 2x2 output)
//         f->addOp(Operation::convolution("c2", "id3", "k2", 
//                                       0, 1, 0, 1,  // no padding, stride 1
//                                       3, 3,        // input 3x3
//                                       2, 2));      // kernel 2x2
//         
//         // Add matrix-column product (2x2 output becomes 2x1)
//         f->addOp(Operation::matrixProduct("p", "c2", "v", 2, 2, 1));
//         
//         // Add inner product (2x1 with itself -> scalar)
//         f->addOp(Operation::innerProduct("f1", "p", "p", 2));
//         
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("id3", {1, 0, 0, 0, 1, 0, 0, 0, 1});  // 3x3 identity
//         f->setValue("k2", {0, 1, 1, 0});                   // 2x2 kernel
//         f->setValue("v", {1, 1});                          // 2-element vector
// 
//         f->compute();
//         f->computeGrad("f1");
//         
//         kernelGrad = new float[4];
//         result = new float[1];
//         f->getGrad("k2", kernelGrad);
//         f->getValue("f1", result);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete[] kernelGrad;
//         delete[] result;
//         delete f;
//     }
// };
// 
// TEST_F(DaftConvolutionGradInnerProductTest, ConvolutionGradInnerProductTestCompute) {
//     EXPECT_EQ(result[0], 2) << "Daft Convolution*InnerProduct value";
//     float kernelGradValues[4] = {4, 2, 2, 4};
//     for (int i = 0; i < 4; i++) {
//         EXPECT_EQ(kernelGrad[i], kernelGradValues[i])
//             << "Daft Convolution*InnerProduct kernel grad";
//     }
// }
// 
// class DaftConvolutionDoubleGradTest : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     float* kernelGrad;
//     float* result;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add 2x2 input matrix and two 2x2 kernels
//         f->addOp(Operation::matrix("id3", 2, 2));
//         f->addOp(Operation::matrix("k2", 2, 2));
//         f->addOp(Operation::matrix("k3", 2, 2));
//         
//         // First convolution with padding and stride 2
//         f->addOp(Operation::convolution("c2", "id3", "k2", 
//                                       1, 2, 1, 2,  // padding 1, stride 2
//                                       2, 2,        // input 2x2
//                                       2, 2));      // kernel 2x2
//         
//         // Second convolution with no padding, stride 1
//         f->addOp(Operation::convolution("f2", "c2", "k3", 
//                                       0, 1, 0, 1,  // no padding, stride 1
//                                       2, 2,        // c2 output is 2x2
//                                       2, 2));      // kernel 2x2
//         
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("id3", {0, 1, -1, 0});
//         f->setValue("k2", {5, 6, 9, 3});
//         f->setValue("k3", {1, 1, 1, 1});
// 
//         f->compute();
//         f->computeGrad("f2");
//         
//         kernelGrad = new float[4];
//         result = new float[1];
//         f->getGrad("k2", kernelGrad);
//         f->getValue("f2", result);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete[] kernelGrad;
//         delete[] result;
//         delete f;
//     }
// };
// 
// TEST_F(DaftConvolutionDoubleGradTest, ConvolutionDoubleGradTestCompute) {
//     float kernelGradValues[4] = {0, -1, 1, 0};
//     for (int i = 0; i < 4; i++) {
//         EXPECT_EQ(kernelGrad[i], kernelGradValues[i])
//             << "Daft Convolution*Convolution kernel grad";
//     }
// }
// 
// class DaftMaxPoolComputeTest : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     float* result;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add 2x2 input matrix
//         f->addOp(Operation::matrix("id3", 2, 2));
//         
//         // Add MaxPool operation with 2x2 pool size and stride 1
//         f->addOp(Operation::maxPool("mp", "id3", 2, 2, 1, 1, 2, 2));
//         
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("id3", {1, 2, 3, 4});
// 
//         f->compute();
//         
//         result = new float[1];  // 1x1 output
//         f->getValue("mp", result);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete[] result;
//         delete f;
//     }
// };
// 
// TEST_F(DaftMaxPoolComputeTest, MaxPoolComputeTest) {
//     EXPECT_EQ(result[0], 4) << "Daft MaxPool compute";
// }
// 
// class DaftMaxPoolLargeTest : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     float* result;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add 4x4 input matrix
//         f->addOp(Operation::matrix("id3", 4, 4));
//         
//         // Add MaxPool operation with 2x2 pool size and stride 2
//         f->addOp(Operation::maxPool("mp", "id3", 2, 2, 2, 2, 4, 4));
//         
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("id3", {1, 2, 1, 2, 3, 9, 16, 3, 1, 10, 4, 1, 3, 4, 2, 3});
// 
//         f->compute();
//         
//         result = new float[4];  // 2x2 output
//         f->getValue("mp", result);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete[] result;
//         delete f;
//     }
// };
// 
// TEST_F(DaftMaxPoolLargeTest, MaxPoolLargeTest) {
//     float values[4] = {9, 16, 10, 4};
//     for (int i = 0; i < 4; i++) {
//         EXPECT_EQ(result[i], values[i]) << "Daft MaxPool large test";
//     }
// }
// 
// class DaftMaxPoolGradTest : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     vector<vector<float>> testvalue;
//     float scalarValue = 5;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add 2x2 input matrix
//         f->addOp(Operation::matrix("id3", 2, 2));
//         
//         // Add MaxPool operation with 2x2 pool size and stride 1
//         f->addOp(Operation::maxPool("mp", "id3", 2, 2, 1, 1, 2, 2));
//         
//         // Add scalar multiplication
//         f->addOp(Operation::scalarMultiply("smp", "mp", 1, 1, scalarValue));
//         
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("id3", {1, 1, 1, 4});
// 
//         f->compute();
//         f->computeGrad("smp");
//         
//         f->getGrad("id3", &testvalue);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete f;
//     }
// };
// 
// TEST_F(DaftMaxPoolGradTest, MaxPoolGradTest) {
//     float values[4] = {0, 0, 0, scalarValue};
//     for (int i = 0; i < 4; i++) {
//         EXPECT_EQ(testvalue[0][i], values[i]) << "Daft MaxPool grad test";
//     }
// }
// 
// class DaftConcatComputeTest : public testing::Test {
// protected:
//     cublasHandle_t cublasH;
//     Function* f;
//     vector<vector<float>> result;
//     vector<vector<float>> testgrad;
// 
//     void SetUp() override {
//         cublasCreate(&cublasH);
//         f = new Function(&cublasH);
//         
//         // Add two single-element columns and one 2-element column
//         f->addOp(Operation::column("v2", 1));
//         f->addOp(Operation::column("v3", 1));
//         f->addOp(Operation::column("v4", 2));
//         
//         // Add concat operation to combine v2 and v3 into a 2-element vector
//         f->addOp(Operation::concat("v2v3", {"v2", "v3"}, 2));
//         
//         // Add inner product between the concatenated vector and v4
//         f->addOp(Operation::innerProduct("flatF", "v2v3", "v4", 2));
//         
//         f->compile();
// 
//         // Set up values - same as silly test
//         f->setValue("v2", {1});
//         f->setValue("v3", {2});
//         f->setValue("v4", {3, 5});
// 
//         f->compute();
//         f->computeGrad("flatF");
//         
//         f->getValue("flatF", &result);
//         f->getGrad("v2", &testgrad);
//     }
//     
//     void TearDown() override {
//         cublasDestroy(cublasH);
//         delete f;
//     }
// };
// 
// TEST_F(DaftConcatComputeTest, ConcatComputeTest) {
//     EXPECT_EQ(result[0][0], 13) << "Daft Concat compute";
//     EXPECT_EQ(testgrad[0][0], 3) << "Daft Concat grad";
// }

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
        f->setValue("A2", {{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}});  // Permutation matrix

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

class DaftBatchNegativeIdentityRegularTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add 2x2 negative identity matrix and 2x1 column vector
        f->addOp(Operation::matrix("negId", 2, 2));
        f->addOp(Operation::column("x", 2));
        
        // Multiply negative identity by vector
        f->addOp(Operation::matrixProduct("result", "negId", "x", 2, 2, 1));
        f->compile(10);  // Compile for batch size of 10

        // Set negative identity matrix: [[-1, 0], [0, -1]]
        // Note: InputMatrix only needs one copy since it's special
        f->setValue("negId", {{-1, 0, 0, -1}});
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchNegativeIdentityRegularTest, BatchNegativeIdentityRegularTest) {
    // Set 10 different 2D vectors for batch processing
    vector<vector<float>> inputVectors = {
        {1.5, 2.3},    // batch 0
        {-0.7, 4.1},   // batch 1
        {3.2, -1.8},   // batch 2
        {0.0, 5.5},    // batch 3
        {-2.4, -3.1},  // batch 4
        {7.7, 0.9},    // batch 5
        {-5.0, 2.0},   // batch 6
        {1.1, -4.4},   // batch 7
        {6.6, 8.8},    // batch 8
        {-9.9, -0.1}   // batch 9
    };

    f->setValue("x", inputVectors);
    f->compute();

    // Get results and verify each vector becomes its negative
    vector<vector<float>> results;
    f->getValue("result", &results);

    vector<vector<float>> expected = {
        {-1.5, -2.3},    // batch 0
        {0.7, -4.1},     // batch 1
        {-3.2, 1.8},     // batch 2
        {0.0, -5.5},     // batch 3
        {2.4, 3.1},      // batch 4
        {-7.7, -0.9},    // batch 5
        {5.0, -2.0},     // batch 6
        {-1.1, 4.4},     // batch 7
        {-6.6, -8.8},    // batch 8
        {9.9, 0.1}       // batch 9
    };

    for (int batch = 0; batch < 10; batch++) {
        EXPECT_FLOAT_EQ(results[batch][0], expected[batch][0]) 
            << "Regular batch negative identity batch " << batch << ", element 0";
        EXPECT_FLOAT_EQ(results[batch][1], expected[batch][1]) 
            << "Regular batch negative identity batch " << batch << ", element 1";
    }
}
