#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>  // 支持 Eigen 与 NumPy 互操作
#include <Eigen/Dense>

namespace py = pybind11;

// 使用 Eigen 实现 SVD
Eigen::MatrixXd svd_eigen(const Eigen::MatrixXd& matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix,
        Eigen::ComputeThinU | Eigen::ComputeThinV
    );
    return svd.singularValues();
}

// PyBind11 绑定
PYBIND11_MODULE(svd_eigen, m) {
    m.def("svd", &svd_eigen, "Compute SVD using Eigen");
}