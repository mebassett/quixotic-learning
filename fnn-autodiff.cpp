#include <iostream>
import autodiff;
    
using namespace std;
const unsigned int INPUT_SIZE = 784;
const unsigned int OUTPUT_SIZE = 10;

ADV_Vec input ("input", INPUT_SIZE);
ADV_Vec weights ("weights0", INPUT_SIZE+1);



int main() {
    cout << "hello, world!\n";
    return 0;
}
