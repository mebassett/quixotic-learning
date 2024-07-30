// mnistdata.h
#ifndef MNISTDATA_H
#define MNISTDATA_H
#include <iostream>
#include<valarray>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
using namespace std;

namespace MNIST {

const unsigned int INPUT_SIZE = 784;
const unsigned int OUTPUT_SIZE = 10;

struct Training_Datum {
  valarray<double> x;
  int y;
  valarray<double> t;
};
typedef vector<Training_Datum> Training_Data ;

valarray<double> to_model_output(int in) ;
Training_Data load_data_from_file(string filename, int cutoff) ;
}
#endif /* MNISTDATA_H */
