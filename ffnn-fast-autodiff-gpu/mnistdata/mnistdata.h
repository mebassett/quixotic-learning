// mnistdata.h
#ifndef MNISTDATA_H
#define MNISTDATA_H
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <valarray>
#include <vector>
using namespace std;

namespace MNIST {

const unsigned int INPUT_SIZE = 784;
const unsigned int OUTPUT_SIZE = 10;

struct Training_Datum {
  valarray<float> x;
  int y;
  valarray<float> t;
};
typedef vector<Training_Datum> Training_Data;

valarray<float> to_model_output(int in);
Training_Data load_data_from_file(string filename, int cutoff);
Training_Data load_data_from_file_no_bias(string filename, int cutoff);
} // namespace MNIST
#endif /* MNISTDATA_H */
