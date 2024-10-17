#include "../silly_autodiff/silly_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <valarray>

using namespace FA;

class ScalarTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Col* xy;
    Scalar* test_scalar;
    void SetUp() override
    {
        cublasCreate(&cublasH);

        this->xy = new Col("xy", 2);
        this->test_scalar = new Scalar(xy, 5);
    }
    void TearDown() override
    {
        delete this->test_scalar;
        cublasDestroy(cublasH);
    }
};

TEST_F(ScalarTest, ScalarCompute)
{
    xy->loadValues({ 1, 2 });
    test_scalar->compute(&cublasH);
    test_scalar->fromDevice();
    EXPECT_EQ(test_scalar->value[0], 5);
    EXPECT_EQ(test_scalar->value[1], 10);
}

class InnerProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;

    Col* ab;
    Col* xy;
    Col* x;
    InnerProduct* test_ip;
    InnerProduct* f;
    float* grad;
    void SetUp() override
    {
        cublasCreate(&cublasH);
        ab = new Col("ab", 2);
        xy = new Col("xy", 2);
        this->test_ip = new InnerProduct(xy, ab);
        x = new Col("x", 1);

        f = new InnerProduct(x, x);
        grad = new float;
    }
    void TearDown() override
    {
        delete test_ip;
        delete f;
        delete grad;
        cublasDestroy(cublasH);
    }
};

TEST_F(InnerProductTest, InnerProductCompute)
{
    ab->loadValues({ 3.0, 4.0 });
    xy->loadValues({ 1.0, 2.0 });
    test_ip->compute(&cublasH);
    test_ip->fromDevice();
    EXPECT_EQ(test_ip->value[0], 11.0) << "compute";

    x->loadValues({ 9 });
    f->compute(&cublasH);
    f->computeGrad(&cublasH);
    cudaMemcpy(grad, x->d_grad, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(*grad, 18) << "x grad";
}

class MatrixColProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    float* matrixGrad;
    Matrix* abcd;
    Col* xy;
    MatrixColProduct* test_matCol;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        abcd = new Matrix("abcd", 2, 2);
        xy = new Col("xy", 2);
        test_matCol = new MatrixColProduct(abcd, xy);
        matrixGrad = new float[4];

        abcd->loadValues({ 1, -1, -1, 1 });
        xy->loadValues({ 1, 2 });

        test_matCol->compute(&cublasH);
        test_matCol->fromDevice();
        test_matCol->computeGrad(&cublasH);
        cudaMemcpy(matrixGrad, abcd->d_grad, 4 * sizeof(float),
            cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete[] matrixGrad;
        delete test_matCol;
    }
};

TEST_F(MatrixColProductTest, MatrixColProductCompute)
{

    EXPECT_EQ(test_matCol->value[0], -1) << "compute";
    EXPECT_EQ(test_matCol->value[1], 1) << "compute";
    EXPECT_EQ(matrixGrad[0], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[1], 2) << "abcd grad";
    EXPECT_EQ(matrixGrad[2], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[3], 2) << "abcd grad";
}

class LeakyReLUTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Col* z;
    ColLeakyReLU* relu;
    float* matrixGrad;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        z = new Col("z", 4);
        relu = new ColLeakyReLU(z);
        matrixGrad = new float[4];

        z->loadValues({ 500, -500, 0.5, -1 });
        relu->compute(&cublasH);
        relu->computeGrad(&cublasH);
        relu->fromDevice();

        cudaMemcpy(matrixGrad, z->d_grad, 4 * sizeof(float),
            cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete[] matrixGrad;
        delete relu;
    }
};

TEST_F(LeakyReLUTest, LeakyReLUComputeAndGrad)
{
    float values[4] = { 500, -5, 0.5, -0.01 };
    float grads[4] = { 1, 0.01, 1, 0.01 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(relu->value[i], values[i]) << "LeakyReLU compute";
        EXPECT_EQ(matrixGrad[i], grads[i]) << "z grad";
    }
}

class ConvolutionTestNoPaddingSingleOffset : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix *inputValues, *kernel;
    Convolution* conv;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        inputValues = new Matrix("input", 3, 3);
        kernel = new Matrix("kernel", 2, 2);
        conv = new Convolution(inputValues, kernel, 0, 1, 0, 1);

        inputValues->loadValues({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        kernel->loadValues({ 3, 3, 3, 3 });

        conv->compute(&cublasH);
        conv->fromDevice();
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete conv;
    }
};

TEST_F(ConvolutionTestNoPaddingSingleOffset, ConvolutionTestCompute)
{
    float values[4] = { 36, 48, 72, 84 };
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(conv->value[i], values[i])
            << "Convolution compute, no padding single offset.";
}

class ConvolutionTestPaddedWithStride : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix *inputValues, *kernel;
    Convolution* conv;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        inputValues = new Matrix("input", 4, 4);
        kernel = new Matrix("kernel", 3, 3);
        conv = new Convolution(inputValues, kernel, 1, 3, 1, 3);

        inputValues->loadValues({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7 });
        kernel->loadValues({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });

        conv->compute(&cublasH);
        conv->fromDevice();
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete conv;
    }
};

TEST_F(ConvolutionTestPaddedWithStride, ConvolutionTestCompute)
{
    float values[4] = { 7, 4, 4, 9 };
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(conv->value[i], values[i])
            << "Convolution compute, 1 padding, 3 stride.";
}

class ConvolutionGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix *inputValues, *kernel;
    Convolution* conv;
    float* kernelGrad;
    float* inputGrad;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        inputValues = new Matrix("input", 2, 2);
        kernel = new Matrix("kernel", 2, 2);
        conv = new Convolution(inputValues, kernel, 0, 1, 0, 1);

        inputValues->loadValues({ 1, 2, 3, 4 });
        kernel->loadValues({ 3, 3, 3, 3 });

        conv->compute(&cublasH);
        conv->computeGrad(&cublasH);

        kernelGrad = new float[4];
        inputGrad = new float[4];
        cudaMemcpy(kernelGrad, kernel->d_grad, sizeof(float) * 4,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(inputGrad, inputValues->d_grad, sizeof(float) * 4,
            cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete[] kernelGrad;
        delete[] inputGrad;
        delete conv;
    }
};

TEST_F(ConvolutionGradTest, ConvolutionTestCompute)
{
    float kernelGradValues[4] = { 1, 2, 3, 4 };
    float inputGradValues[4] = { 3, 3, 3, 3 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[i], kernelGradValues[i]) << "Convolution kernel grad";
        EXPECT_EQ(inputGrad[i], inputGradValues[i]) << "Convolution kernel grad";
    }
}

class ConvolutionGradInnerProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix *id3, *k2;
    Col* v;
    Convolution* c2;
    MatrixColProduct* p;
    InnerProduct* f1;
    float* kernelGrad;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        id3 = new Matrix("id3", 3, 3);
        k2 = new Matrix("k2", 2, 2);
        v = new Col("v", 2);
        c2 = new Convolution(id3, k2, 0, 1, 0, 1);
        p = new MatrixColProduct(c2, v);
        f1 = new InnerProduct(p, p);

        id3->loadValues({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });
        k2->loadValues({ 0, 1, 1, 0 });
        v->loadValues({ 1, 1 });

        f1->compute(&cublasH);
        f1->computeGrad(&cublasH);
        f1->fromDevice();

        kernelGrad = new float[4];
        cudaMemcpy(kernelGrad, k2->d_grad, sizeof(float) * 4,
            cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete[] kernelGrad;
        delete f1;
    }
};

TEST_F(ConvolutionGradInnerProductTest, ConvolutionTestCompute)
{
    EXPECT_EQ(f1->value[0], 2) << "Convolution*InnerProduct value";
    float kernelGradValues[4] = { 4, 2, 2, 4 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[i], kernelGradValues[i])
            << "Convolution*InnerProduct kernel grad";
    }
}

class ConvolutionDoubleGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix *id3, *k2, *k3;
    Convolution *c2, *f2;
    float* kernelGrad;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        id3 = new Matrix("id3", 2, 2);
        k2 = new Matrix("k2", 2, 2);
        k3 = new Matrix("k3", 2, 2);
        c2 = new Convolution(id3, k2, 1, 2, 1, 2);
        f2 = new Convolution(c2, k3, 0, 1, 0, 1);

        id3->loadValues({ 0, 1, -1, 0 });
        k2->loadValues({ 5, 6, 9, 3 });
        k3->loadValues({ 1, 1, 1, 1 });

        f2->compute(&cublasH);
        f2->computeGrad(&cublasH);
        f2->fromDevice();

        kernelGrad = new float[4];
        cudaMemcpy(kernelGrad, k2->d_grad, sizeof(float) * 4,
            cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete[] kernelGrad;
        delete f2;
    }
};

TEST_F(ConvolutionDoubleGradTest, ConvolutionTestCompute)
{
    float kernelGradValues[4] = { 0, -1, 1, 0 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(kernelGrad[i], kernelGradValues[i])
            << "Convolution*InnerProduct kernel grad";
    }
}

class MaxPoolComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix* m;
    MaxPool* mp;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        m = new Matrix("id3", 2, 2);
        mp = new MaxPool(m, 2, 2, 1, 1);

        m->loadValues({ 1, 2, 3, 4 });

        mp->compute(&cublasH);
        mp->fromDevice();
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete mp;
    }
};

TEST_F(MaxPoolComputeTest, MaxPoolComputetest) { EXPECT_EQ(mp->value[0], 4); }

class MaxPoolLargeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix* m;
    MaxPool* mp;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        m = new Matrix("id3", 4, 4);
        mp = new MaxPool(m, 2, 2, 2, 2);

        m->loadValues({ 1, 2, 1, 2, 3, 9, 16, 3, 1, 10, 4, 1, 3, 4, 2, 3 });

        mp->compute(&cublasH);
        mp->fromDevice();
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete mp;
    }
};

TEST_F(MaxPoolLargeTest, MaxPoolLargeTest)
{
    float values[4] = { 9, 16, 10, 4 };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(mp->value[i], values[i]);
    }
}

class MaxPoolGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix* m;
    MaxPool* mp;
    Scalar* smp;
    float* testvalue;
    float scalarValue = 5;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        m = new Matrix("id3", 2, 2);
        mp = new MaxPool(m, 2, 2, 1, 1);
        smp = new Scalar(mp, scalarValue);
        testvalue = new float[4];

        m->loadValues({ 1, 1, 1, 4 });

        smp->compute(&cublasH);
        smp->computeGrad(&cublasH);
        cudaMemcpy(testvalue, m->d_grad, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete smp;
        delete[] testvalue;
    }
};

TEST_F(MaxPoolGradTest, MaxPoolLargeTest)
{
    float values[4] = { 0, 0, 0, scalarValue };
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(testvalue[i], values[i]);
    }
}

class FlattenComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix* m;
    Flatten* flat;
    InnerProduct* flatF;

    void SetUp() override
    {
        cublasCreate(&cublasH);

        m = new Matrix("m-3", 2, 2);
        flat = new Flatten(m);
        flatF = new InnerProduct(flat, flat);

        m->loadValues({ 1, 2, 3, 4 });

        flatF->compute(&cublasH);
        flatF->fromDevice();
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete flatF;
    }
};

TEST_F(FlattenComputeTest, FlattenComputeTest)
{
    EXPECT_EQ(flatF->value[0], 1 + 4 + 9 + 16);
}

class FlattenGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Matrix *m, *k4;
    Col* v1;
    Convolution* c1;
    Flatten* flat;
    InnerProduct* flatF;
    float* testvalue;

    void SetUp() override
    {
        cublasCreate(&cublasH);

        m = new Matrix("m-4", 3, 3);
        k4 = new Matrix("k4", 2, 2);
        v1 = new Col("v1", 4);
        c1 = new Convolution(m, k4, 0, 1, 0, 1);
        flat = new Flatten(c1);
        flatF = new InnerProduct(flat, v1);

        m->loadValues({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        k4->loadValues({ 1, 1, 1, 1 });
        v1->loadValues({ 1, 2, 3, 4 });

        flatF->compute(&cublasH);
        flatF->computeGrad(&cublasH);
        testvalue = new float[9];
        cudaMemcpy(testvalue, m->d_grad, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete flatF;
        delete[] testvalue;
    }
};

TEST_F(FlattenGradTest, FlattenGradTest)
{
    float values[9] = { 1, 3, 2, 4, 10, 6, 3, 7, 4 };
    for (int i = 0; i < 9; i++) {
        EXPECT_EQ(testvalue[i], values[i]);
    }
}

class ConcatComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Col *v2, *v3, *v4;
    AD* v2v3;
    InnerProduct* flatF;
    float* testgrad;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        v2 = new Col("v2", 1);
        v3 = new Col("v3", 1);
        v4 = new Col("v4", 2);
        v2v3 = new ConcatCol({ v2, v3 });
        flatF = new InnerProduct(v2v3, v4);

        v2->loadValues({ 1 });
        v3->loadValues({ 2 });
        v4->loadValues({ 3, 5 });

        flatF->compute(&cublasH);
        flatF->computeGrad(&cublasH);
        flatF->fromDevice();

        testgrad = new float;
        cudaMemcpy(testgrad, v2->d_grad, sizeof(float), cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete flatF;
        delete testgrad;
    }
};

TEST_F(ConcatComputeTest, ConcatComputeTest)
{
    EXPECT_EQ(flatF->value[0], 13) << "compute";
    EXPECT_EQ(testgrad[0], 3) << "grad";
}

class ConcatFlattenComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Col *v21, *v31, *v41, *v51;
    Flatten *ip1, *ip2;
    AD* v2v3;
    Col* v6;
    InnerProduct* flatF;
    float* testgrad;

    void SetUp() override
    {
        cublasCreate(&cublasH);
        v21 = new Col("v2-1", 2);
        v31 = new Col("v3-1", 2);
        v41 = new Col("v4-1", 2);
        v51 = new Col("v5-1", 2);
        ip1 = new Flatten(new InnerProduct(v41, v21));
        ip2 = new Flatten(new InnerProduct(v51, v31));
        v2v3 = new ConcatCol({ ip1, ip2 });
        v6 = new Col("v6", 2);
        flatF = new InnerProduct(v2v3, v6);

        v21->loadValues({ 2, 2 });
        v31->loadValues({ 3, 3 });
        v41->loadValues({ 1, 1 });
        v51->loadValues({ 1, 1 });
        v6->loadValues({ 3, 4 });

        flatF->compute(&cublasH);
        flatF->fromDevice();
        flatF->computeGrad(&cublasH);

        testgrad = new float[2];
        cudaMemcpy(testgrad, v21->d_grad, 2 * sizeof(float),
            cudaMemcpyDeviceToHost);
    }
    void TearDown() override
    {
        cublasDestroy(cublasH);
        delete flatF;
        delete[] testgrad;
    }
};

TEST_F(ConcatFlattenComputeTest, ConcatFlattenComputeTest)
{
    EXPECT_EQ(flatF->value[0], 36) << "compute";
    EXPECT_EQ(testgrad[0], 3) << "grad";
    EXPECT_EQ(testgrad[1], 3) << "grad";
}
