#include "film_update.h"
#include <pybind11/iostream.h>  // 用于重定向输出
#include <iostream>
#include <pybind11/eigen.h>  // 支持 Eigen 与 NumPy 互操作
#include <Eigen/Dense>

// 基础打印函数
void print_values(int a, double b) {
    std::cout << "C++ 打印的值: a = " << a << ", b = " << b << std::endl;
}

// 重定向输出到 Python 的 sys.stdout
void redirect_output() {
    py::scoped_ostream_redirect stream(
        std::cout,
        py::module_::import("sys").attr("stdout")
    );
    std::cout << "此输出已重定向到 Python 的 sys.stdout" << std::endl;
}


// 使用 Eigen 实现 SVD
Eigen::MatrixXd svd_eigen(const Eigen::MatrixXd& matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix,
        Eigen::ComputeThinU | Eigen::ComputeThinV
    );
    return svd.singularValues();
}



py::array_t<double> get_normal_from_grid(
    py::array_t<int> film,
    py::array_t<double> normal_matrix,
    int mirrorGap,
    py::array_t<int> point
) {
    // 获取原始指针和维度信息
    auto film_buf = film.request();
    int* film_ptr = static_cast<int*>(film_buf.ptr);
    const int dim_x = film_buf.shape[0];
    const int dim_y = film_buf.shape[1];
    const int dim_z = film_buf.shape[2];

    auto normal_buf = normal_matrix.request();
    double* normal_ptr = static_cast<double*>(normal_buf.ptr);

    auto point_buf = point.request();
    int* point_ptr = static_cast<int*>(point_buf.ptr);
    int x = point_ptr[0] + mirrorGap;
    int y = point_ptr[1] + mirrorGap;
    int z = point_ptr[2];

    // 三维索引计算函数
    auto film_index = [dim_y, dim_z](int x, int y, int z) {
        return x * dim_y * dim_z + y * dim_z + z;
    };

    auto normal_index = [dim_y, dim_z](int x, int y, int z) {
        return 3*(x * dim_y * dim_z + y * dim_z + z);
    };

    // 步骤1: 收集7x7x7立方体内值为1的坐标
    std::vector<Eigen::Vector3d> positions;
    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dz = -3; dz <= 3; ++dz) {
                int xi = x + dx;
                int yi = y + dy;
                int zi = z + dz;
                
                if (xi >= 0 && xi < dim_x && 
                    yi >= 0 && yi < dim_y && 
                    zi >= 0 && zi < dim_z) 
                {
                    // 直接通过内存指针访问
                    if (film_ptr[film_index(xi, yi, zi)] == 1) {
                        positions.emplace_back(dx, dy, dz);
                    }
                }
            }
        }
    }

    // ... [保持原有计算逻辑不变]
    // 如果没有有效点，返回原矩阵
    if (positions.empty()) {
        return normal_matrix;
    }

    // 步骤2: 计算均值
    Eigen::Vector3d mean(0, 0, 0);
    for (const auto& pos : positions) {
        mean += pos;
    }
    mean /= positions.size();

    // 步骤3: 计算协方差矩阵
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pos : positions) {
        Eigen::Vector3d centered = pos - mean;
        cov += centered * centered.transpose();
    }
    // 步骤4: SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
    Eigen::Vector3d normal = svd.matrixU().col(2); // 最小特征值对应的特征向量
    // 步骤5: 写入法线数据
    x -= mirrorGap; // 恢复原始坐标
    y -= mirrorGap;
    
    if (x >= 0 && x < dim_x && 
        y >= 0 && y < dim_y && 
        z >= 0 && z < dim_z) 
    {
        const size_t base_idx = normal_index(x, y, z);
        if (z + 2 < dim_z) {
            normal_ptr[base_idx]     = normal[0];
            normal_ptr[base_idx + 1] = normal[1];
            normal_ptr[base_idx + 2] = normal[2];
        } else {
            // 处理边界情况
            const int available = dim_z - z;
            for (int i = 0; i < available; ++i) {
                normal_ptr[base_idx + i] = normal[i];
            }
        }
    }

    return normal_matrix;
}


void get_normal_from_grid_Cell(
    py::array_t<Cell, py::array::c_style> cell_array,
    // py::array_t<int> film,
    // py::array_t<double> normal_matrix,
    // int mirrorGap,
    py::array_t<int> point
) {
    // 获取原始指针和维度信息
    auto cell_buf = cell_array.request();
    auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

    // auto film_buf = film.request();
    // int* film_ptr = static_cast<int*>(film_buf.ptr);
    const int dim_x = cell_array.shape(0);
    const int dim_y = cell_array.shape(0);
    const int dim_z = cell_array.shape(0);

    // auto normal_buf = normal_matrix.request();
    // double* normal_ptr = static_cast<double*>(normal_buf.ptr);

    auto point_buf = point.request();
    int* point_ptr = static_cast<int*>(point_buf.ptr);
    int x = point_ptr[0];
    int y = point_ptr[1];
    int z = point_ptr[2];

    // 三维索引计算函数
    auto film_index = [dim_y, dim_z](int x, int y, int z) {
        return x * dim_y * dim_z + y * dim_z + z;
    };

    auto normal_index = [dim_y, dim_z](int x, int y, int z) {
        return x * dim_y * dim_z + y * dim_z + z;
    };

    // 步骤1: 收集7x7x7立方体内值为1的坐标
    std::vector<Eigen::Vector3d> positions;
    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dz = -3; dz <= 3; ++dz) {
                int xi = x + dx;
                int yi = y + dy;
                int zi = z + dz;
                
                if (xi >= 0 && xi < dim_x && 
                    yi >= 0 && yi < dim_y && 
                    zi >= 0 && zi < dim_z) 
                {
                    // 直接通过内存指针访问
                    if (cell_ptr[film_index(xi, yi, zi)].id == 1) {
                        positions.emplace_back(dx, dy, dz);
                    }
                }
            }
        }
    }

    // ... [保持原有计算逻辑不变]
    // 如果没有有效点，返回原矩阵
    if (positions.empty()) {
        std::cout << "svd矩阵为空" << std::endl;
    }

    // 步骤2: 计算均值
    Eigen::Vector3d mean(0, 0, 0);
    for (const auto& pos : positions) {
        mean += pos;
    }
    mean /= positions.size();

    // 步骤3: 计算协方差矩阵
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pos : positions) {
        Eigen::Vector3d centered = pos - mean;
        cov += centered * centered.transpose();
    }

    // 步骤4: SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
    Eigen::Vector3d normal = svd.matrixU().col(2); // 最小特征值对应的特征向量

    // 步骤5: 写入法线数据
    if (x >= 0 && x < dim_x && 
        y >= 0 && y < dim_y && 
        z >= 0 && z < dim_z) 
    {
        const size_t base_idx = normal_index(x, y, z);
        if (z + 2 < dim_z) {
            cell_ptr[base_idx].normal[0] = normal[0];
            cell_ptr[base_idx].normal[1] = normal[1];
            cell_ptr[base_idx].normal[2] = normal[2];
        } else {
            // 处理边界情况
            const int available = dim_z - z;
            for (int i = 0; i < available; ++i) {
                cell_ptr[base_idx].normal[i] = normal[i];
            }
        }
    }
}


void get_normal_from_grid_Cell_toc(
    py::array_t<Cell, py::array::c_style> cell_array,
    // py::array_t<int> film,
    // py::array_t<double> normal_matrix,
    // int mirrorGap,
    // std:vector<int> point
    int x,
    int y,
    int z
) {
    // 获取原始指针和维度信息
    auto cell_buf = cell_array.request();
    auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

    // auto film_buf = film.request();
    // int* film_ptr = static_cast<int*>(film_buf.ptr);
    const int dim_x = cell_array.shape(0);
    const int dim_y = cell_array.shape(0);
    const int dim_z = cell_array.shape(0);

    // auto normal_buf = normal_matrix.request();
    // double* normal_ptr = static_cast<double*>(normal_buf.ptr);

    // auto point_buf = point.request();
    // int* point_ptr = static_cast<int*>(point_buf.ptr);
    // int x = point_ptr[0];
    // int y = point_ptr[1];
    // int z = point_ptr[2];

    // 三维索引计算函数
    auto film_index = [dim_y, dim_z](int x, int y, int z) {
        return x * dim_y * dim_z + y * dim_z + z;
    };

    auto normal_index = [dim_y, dim_z](int x, int y, int z) {
        return x * dim_y * dim_z + y * dim_z + z;
    };

    // 步骤1: 收集7x7x7立方体内值为1的坐标
    std::vector<Eigen::Vector3d> positions;
    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dz = -3; dz <= 3; ++dz) {
                int xi = x + dx;
                int yi = y + dy;
                int zi = z + dz;
                
                if (xi >= 0 && xi < dim_x && 
                    yi >= 0 && yi < dim_y && 
                    zi >= 0 && zi < dim_z) 
                {
                    // 直接通过内存指针访问
                    if (cell_ptr[film_index(xi, yi, zi)].id == 1) {
                        positions.emplace_back(dx, dy, dz);
                    }
                }
            }
        }
    }

    // ... [保持原有计算逻辑不变]
    // 如果没有有效点，返回原矩阵
    if (positions.empty()) {
        std::cout << "svd矩阵为空" << std::endl;
    }

    // 步骤2: 计算均值
    Eigen::Vector3d mean(0, 0, 0);
    for (const auto& pos : positions) {
        mean += pos;
    }
    mean /= positions.size();

    // 步骤3: 计算协方差矩阵
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pos : positions) {
        Eigen::Vector3d centered = pos - mean;
        cov += centered * centered.transpose();
    }

    // 步骤4: SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
    Eigen::Vector3d normal = svd.matrixU().col(2); // 最小特征值对应的特征向量

    // 步骤5: 写入法线数据
    if (x >= 0 && x < dim_x && 
        y >= 0 && y < dim_y && 
        z >= 0 && z < dim_z) 
    {
        const size_t base_idx = normal_index(x, y, z);
        if (z + 2 < dim_z) {
            cell_ptr[base_idx].normal[0] = normal[0];
            cell_ptr[base_idx].normal[1] = normal[1];
            cell_ptr[base_idx].normal[2] = normal[2];
        } else {
            // 处理边界情况
            const int available = dim_z - z;
            for (int i = 0; i < available; ++i) {
                cell_ptr[base_idx].normal[i] = normal[i];
            }
        }
    }
}


void update_normal_in_matrix(
    py::array_t<Cell, py::array::c_style> cell_array,
    py::array_t<int, py::array::c_style> point_to_change
) {
    // 获取输入数组信息
    auto cell_buf = cell_array.request();
    auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

    const int dim_x = cell_array.shape(0);
    const int dim_y = cell_array.shape(0);
    const int dim_z = cell_array.shape(0);

    // auto film_mirror_buf = film_label_mirror.request();
    // auto* film_mirror = static_cast<LabeledNormal*>(film_mirror_buf.ptr);
    // const int dim_x = film_mirror_buf.shape[0];
    // const int dim_y = film_mirror_buf.shape[1];
    // const int dim_z = film_mirror_buf.shape[2];

    // auto film_normal_buf = film_label_normal.request();
    // auto* film_normal = static_cast<double*>(film_normal_buf.ptr);

    auto points_buf = point_to_change.request();
    auto* points = static_cast<int*>(points_buf.ptr);
    const int num_points = point_to_change.shape(0);

    // 三维索引计算函数
    auto film_index = [dim_y, dim_z](int x, int y, int z) {
        return x * dim_y * dim_z + y * dim_z + z;
    };

    // 步骤1: 收集所有需要处理的点
    // std::unordered_set<std::tuple<int, int, int>> unique_points;
    std::vector<Eigen::Vector3d> unique_points;
    // #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        int x = points[i * 3];
        int y = points[i * 3 + 1];
        int z = points[i * 3 + 2];

        // 边界检查
        int x_start = std::max(x - 3, 0);
        int x_end = std::min(x + 4, dim_x);
        int y_start = std::max(y - 3, 0);
        int y_end = std::min(y + 4, dim_y);
        int z_start = std::max(z - 3, 0);
        int z_end = std::min(z + 4, dim_z);

        // 遍历7x7x7立方体
        for (int xi = x_start; xi < x_end; ++xi) {
            for (int yi = y_start; yi < y_end; ++yi) {
                for (int zi = z_start; zi < z_end; ++zi) {
                    if (cell_ptr[film_index(xi, yi, zi)].id == 1) {
                        // #pragma omp critical
                        unique_points.emplace_back(xi,yi,zi);
                    }
                }
            }
        }
    }

    #pragma omp parallel for
    for (const auto& pos : unique_points) {
        // 获取唯一点坐标
        int x = pos[0];
        int y = pos[1];
        int z = pos[2];

        // 调用优化后的法线计算函数
        get_normal_from_grid_Cell_toc(
            cell_array,
            x, y, z
        );
    }
    // 步骤2: 去重后更新法线
    // #pragma omp parallel for
    // for(const auto& pos : unique_points) {
    //     // 调用优化后的法线计算函数
    //     get_normal_from_grid_Cell_toc(
    //     cell_array,
    //     pos
    //     );
    // }

    // return film_label_normal;
}




PYBIND11_MODULE(film_optimized, m) {
    using namespace py::literals;

    PYBIND11_NUMPY_DTYPE(Cell, id, index, film, normal);
    // 绑定 Cell 结构体
    // py::class_<Cell>(m, "Cell")
    //     .def(py::init<>())
    //     .def_readwrite("id", &Cell::id)
    //     .def_readwrite("index", &Cell::index)
    //     .def_readwrite("film", &Cell::film)
    //     .def_readwrite("normal", &Cell::normal);

    // 绑定核心函数
    m.def("update_film_label_index_normal_etch", 
        &update_film_label_index_normal_etch,
        "Optimized film update function",
        py::arg("cells"), 
        py::arg("point_etch"), 
        py::arg("cell_size_xyz"));

    m.def("print_values", &print_values, "打印整数和浮点数");

    m.def("redirect_output", &redirect_output, "重定向输出示例");

    m.def("svd", &svd_eigen, "Compute SVD using Eigen");

    m.def("get_normal_from_grid_Cell", &get_normal_from_grid_Cell,
          py::arg("cells"), 
          py::arg("point"));

    m.def("get_normal_from_grid", &get_normal_from_grid,
          "film"_a,
          "normal_matrix"_a,
          "mirrorGap"_a,
          "point"_a);

    m.def("update_normal_in_matrix", &update_normal_in_matrix,
          py::arg("cells"), 
          py::arg("point_to_change"));
}
