#include <iostream>
#include<valarray>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include "mnistdata.h"
using namespace std;

valarray<float> MNIST::to_model_output(int in) {
    valarray<float> ret(OUTPUT_SIZE);
    for(int i = 0; i<OUTPUT_SIZE; i++)
      if(i==in)
        ret[i] = 1;
      else 
        ret[i] = 0;
    return ret;

}

MNIST::Training_Data MNIST::load_data_from_file(string filename, int cutoff) {
  ifstream ifs {filename};

  if(!ifs) {
      cout << "couldn't open training file";
      return {};
  }

  Training_Data rows {};

  for(string row_buffer_str; getline(ifs, row_buffer_str);) {
      istringstream row_buffer;
      valarray<float> current_row(INPUT_SIZE+1);

      row_buffer.str(row_buffer_str);

      int i = 0;
      float max = 0;
      
      for(float f; row_buffer>>f;) {
        if(i<INPUT_SIZE) {
          current_row[i+1] = f;
          if (f > max) max = f;
          i++;
        } else {
          if(max != 0)
              current_row = (1.0/max) * current_row; 
          current_row[0] = 1.0;

          rows.push_back({current_row, (int)lround(f), MNIST::to_model_output((int)lround(f))});
        
          continue;
        }
    
      }
      if(rows.size() >= cutoff) return rows;
  }
  return rows;
}
