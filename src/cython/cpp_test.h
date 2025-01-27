#ifndef TEST_H
#define TEST_H
 
#include "cpp_matrixxfpy.h"
using namespace Eigen; 
 
class Test { 
public: 
    int test1; 
    Test();
    Test(int test1); 
    ~Test(); 
    int returnFour(); 
    int returnFive();
    Test operator+(const Test& other); 
    Test operator-(const Test& other);
    MatrixXfPy getMatrixXf(int d1, int d2);    
};
#endif