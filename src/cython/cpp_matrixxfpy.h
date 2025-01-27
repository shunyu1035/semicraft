#ifndef MATRIXXFPY_H
#define MATRIXXFPY_H
#include <Eigen/Dense>
using namespace Eigen;
 
class MatrixXfPy : public MatrixXf { 
    public: 
        MatrixXfPy() : MatrixXf() { }
        MatrixXfPy(int rows,int cols) : MatrixXf(rows,cols) { }
        MatrixXfPy(const MatrixXf other) : MatrixXf(other) { } 
};
#endif