#include <cublas_v2.h>
#include <vector>
#include <random>

#include "../../daft_autodiff/daft_autodiff.h"
#include "../../mnistdata/mnistdata.h"


using namespace std;
using namespace DA;
using namespace MNIST;

int fromModelOutput(float* out) {
    float max = *max_element(out, out + OUTPUT_SIZE);
    for (int i { 0 }; i < OUTPUT_SIZE; i++) {
        if (*(out + i) == max)
            return i;
    }
    return -1;
}

int main() {
    // see https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf for architecture.
    cout << "Building LeNet...\n" << endl;

}


