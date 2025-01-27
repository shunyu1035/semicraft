#include <Eigen/Dense>
#include <vector>

// 核心计算函数声明
void eigen_compute_svd(
    const double* input, 
    int rows, 
    int cols,
    double* U,
    double* S,
    double* V,
    bool full_matrices = false
);