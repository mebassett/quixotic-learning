module;
#include <iostream>
#include<valarray>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
export module mnistdata;
using namespace std;


const unsigned int INPUT_SIZE = 784;
const unsigned int OUTPUT_SIZE = 10;

export struct Training_Datum {
  valarray<double> x;
  int y;
  valarray<double> t;
};
export typedef vector<Training_Datum> Training_Data ;

export valarray<double> to_model_output(int in) {
    valarray<double> ret(OUTPUT_SIZE);
    for(int i = 0; i<OUTPUT_SIZE; i++)
      if(i==in)
        ret[i] = 1;
      else 
        ret[i] = 0;
    return ret;

}

export Training_Data load_data_from_file(string filename, int cutoff) {
  ifstream ifs {filename};

  if(!ifs) {
      cout << "couldn't open training file";
      return {};
  }

  Training_Data rows {};

  for(string row_buffer_str; getline(ifs, row_buffer_str);) {
      istringstream row_buffer;
      valarray<double> current_row(INPUT_SIZE+1);

      row_buffer.str(row_buffer_str);

      int i = 0;
      double max = 0;
      
      for(double f; row_buffer>>f;) {
        if(i<INPUT_SIZE) {
          current_row[i+1] = f;
          if (f > max) max = f;
          i++;
        } else {
          if(max != 0)
              current_row = (1.0/max) * current_row; 
          current_row[0] = 1.0;

          rows.push_back({current_row, (int)lround(f), to_model_output((int)lround(f))});
        
          continue;
        }
    
      }
      if(rows.size() >= cutoff) return rows;
  }
  return rows;
}
