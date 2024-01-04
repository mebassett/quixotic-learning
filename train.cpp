#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>

using namespace std;

int main(void) {
    ifstream ifs {"mnist_train.txt"};

    if(!ifs) {
        cout << "couldn't open training file";
        return 1;
    }

    vector<vector<float>> rows;

    for(string row_buffer_str; getline(ifs, row_buffer_str);) {
        istringstream row_buffer;
        vector<float> current_row {};

        row_buffer.str(row_buffer_str);
        
        for(float f; row_buffer>>f;)
            current_row.push_back(f);

        cout << "finished a row\n";
        rows.push_back(current_row);
        current_row = {};
    }

    cout << rows.size() << "\n\nall done!\n\n";

    for(const vector<float> row : rows) { 
      for(const float cell : row) 
        cout << cell << "\t";

      cout << "\n";
    }

    return 0;
}
