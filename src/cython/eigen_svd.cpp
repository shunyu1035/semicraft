#include "eigen_svd.h"
#include <Eigen/SVD>

using namespace Eigen;

void eigen_compute_svd(
    const double* input, 
    int rows, 
    int cols,
    double* U,
    double* S,
    double* V,
    bool full_matrices
) {
    // 将输入数据映射为Eigen矩阵
    Map<const MatrixXd> A(input, rows, cols);

    // 选择计算模式
    unsigned int computationOptions = ComputeThinU | ComputeThinV;
    if(full_matrices) {
        computationOptions = ComputeFullU | ComputeFullV;
    }

    // 执行SVD分解
    BDCSVD<MatrixXd> svd(A, computationOptions);

    // 输出结果到缓冲区
    if(U) {
        Map<MatrixXd> U_map(U, svd.matrixU().rows(), svd.matrixU().cols());
        U_map = svd.matrixU();
    }
    
    if(S) {
        Map<VectorXd> S_map(S, svd.singularValues().size());
        S_map = svd.singularValues();
    }

    if(V) {
        Map<MatrixXd> V_map(V, svd.matrixV().rows(), svd.matrixV().cols());
        V_map = svd.matrixV();
    }
}