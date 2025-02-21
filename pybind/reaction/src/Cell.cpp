#include "Cell.h"
#include <pybind11/iostream.h>  // 用于重定向输出
#include <pybind11/eigen.h>  // 支持 Eigen 与 NumPy 互操作
#include <Eigen/Dense>


// Rnd rnd;

void World::WprintCell(int idx, int idy, int idz) {
    std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].typeID: " << Cells[idx][idy][idz].typeID << std::endl;
    std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].index: " << Cells[idx][idy][idz].index << std::endl;    // 输出: particles[id].pos
    std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].normal: " << Cells[idx][idy][idz].normal << std::endl;    // 输出: particles[id].pos
    std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].film: " << std::endl;
    for (size_t i = 0; i < Cells[idx][idy][idz].film.size(); ++i) {
        std::cout << Cells[idx][idy][idz].film[i] << " ";
    }
    std::cout << std::endl;
}


double World::linear_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    // 检查输入数组是否有效
    // if (xp.size() != fp.size() || xp.empty()) {
    // 	throw std::invalid_argument("xp 和 fp 必须有相同且非空的长度。");
    // }
    
    // 边界情况：如果 x 在 xp 的最左侧或最右侧，则直接返回端点值
    if (x <= xp.front()) {
        return fp.front();
    }
    if (x >= xp.back()) {
        return fp.back();
    }
    
    // 使用 std::upper_bound 找到第一个大于 x 的元素
    auto it = std::upper_bound(xp.begin(), xp.end(), x);
    int i = static_cast<int>(it - xp.begin()) - 1;  // xp[i] <= x < xp[i+1]
    
    double x0 = xp[i];
    double x1 = xp[i + 1];
    double y0 = fp[i];
    double y1 = fp[i + 1];
    
    // 线性插值公式
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}


void World::film_add(int3 posInt, std::vector<int> react_add){
    for (int a=0; a<FILMSIZE; ++a){
        // std::cout << react_add[a] <<  std::endl;
        Cells[posInt[0]][posInt[1]][posInt[2]].film[a] += react_add[a];
        // std::cout << Cells[posInt[0]][posInt[1]][posInt[2]].film[a] <<  std::endl;
    }
}


std::vector<int> World::sticking_probability_structed(Particle particle, const Cell cell, double angle_rad, Rnd &rnd) {

    std::vector<int> sticking_acceptList(FILMSIZE, 0);
    std::vector<double> choice(FILMSIZE);

    // 使用外部传入的 rnd 生成 [0,1) 内的随机数
    for (int i = 0; i < FILMSIZE; ++i) {
        choice[i] = rnd();
    }

    int energy_range = 0;
    double sticking_rate = 0.0;
    int particle_id = (particle.id >= 2) ? 2 : particle.id;

    for (int j = 0; j < FILMSIZE; ++j) {
        if (cell.film[j] <= 0) {
            choice[j] = 1.0;
        }

        if (particle_id == ArgonID) {
            energy_range = 0;
            for (int e = 0; e < rn_energy.size(); ++e) {
                if (particle.E < rn_energy[e]) {
                    energy_range = e;
                    // std::cout << "energy_range:  "<< energy_range <<  std::endl;
                    break;

                }
            }
            sticking_rate = linear_interp(angle_rad, rn_angle, rn_matrix[energy_range]);
            // std::cout << "sticking_rate:  "<< sticking_rate <<  std::endl;
        } else if (particle_id < ArgonID) {
            sticking_rate = react_prob_chemical[particle_id][j];
        } else if (particle_id > ArgonID) {
            sticking_rate = react_redepo_sticking[particle_id - ArgonID - 1];
        }

        if (sticking_rate > choice[j]) {
            sticking_acceptList[j] = 1;
            // std::cout << "sticking_acceptList:  "<< j <<  std::endl;
        }
    }
    return sticking_acceptList;
}


void World::update_Cells(){
    size_t update_film_etch_size = update_film_etch.size();
    std::vector<int3> local_point_nn;
    std::vector<int3> local_point_nn_under;
    std::vector<int3> local_point_nn_vaccum;

    if (update_film_etch_size != 0){
        for (size_t i=0; i<update_film_etch_size; i++){
            int3 posInt = update_film_etch[i];

            // std::cout << "posInt: " << posInt << std::endl;

            local_point_nn.resize(6);
            Cells[posInt[0]][posInt[1]][posInt[2]].typeID = -1;
            for (size_t j=0; j<6; ++j){
                local_point_nn[j] = mirror_index(posInt + grid_cross[j]);
                

                // std::cout << "inloop local_point_nn: " << local_point_nn[j] << std::endl;

                if (Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID == 2){
                    Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 1;

                    local_point_nn_under.resize(6);
                    for (size_t k=0; k<6; ++k){
                        local_point_nn_under[k] = mirror_index(local_point_nn[j] + grid_cross[k]);
                        if (Cells[local_point_nn_under[k][0]][local_point_nn_under[k][1]][local_point_nn_under[k][2]].typeID == 3){
                            Cells[local_point_nn_under[k][0]][local_point_nn_under[k][1]][local_point_nn_under[k][2]].typeID = 2;
                        }
                    }
                    // std::cout << "local_point_nn_vaccum: ";
                    // for (size_t q = 0; q < local_point_nn_under.size(); ++q) {
                    //     std::cout << local_point_nn_under[q] << '\n';
                    // }
                    // std::cout << '\n';
                }
                else if (Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID == -1) {

                    local_point_nn_vaccum.resize(6);
                    for (size_t l=0; l<6; ++l){
                        local_point_nn_vaccum[l] = mirror_index(local_point_nn[j] + grid_cross[l]);

                        // std::cout << "local_point_nn_vaccum: " << local_point_nn_vaccum[l] << std::endl;

                        if (Cells[local_point_nn_vaccum[l][0]][local_point_nn_vaccum[l][1]][local_point_nn_vaccum[l][2]].typeID == 1){
                            Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 0;
                        }
                    }

                    // std::cout << "local_point_nn_vaccum: ";
                    // for (size_t f = 0; f < local_point_nn_vaccum.size(); ++f) {
                    //     std::cout << local_point_nn_vaccum[f] << '\n';
                    // }
                    // std::cout << '\n';


                }
            }
            // std::cout << "local_point_nn: ";
            // for (size_t w = 0; w < local_point_nn.size(); ++w) {
            //     std::cout << local_point_nn[w] << '\n';
            // }
            // std::cout << '\n';

        }
    


        update_normal_in_matrix();

        update_film_etch.resize(0);
    }
}





void World::get_normal_from_grid(int3 posInt) {

    // 步骤1: 收集7x7x7立方体内值为1的坐标
    std::vector<Eigen::Vector3d> positions;
    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dz = -3; dz <= 3; ++dz) {
                int xi = posInt[0] + dx;
                int yi = posInt[1] + dy;
                int zi = posInt[2] + dz;
                int3 point = {xi, yi, zi};
                point = mirror_index(point);
                if (Cells[point[0]][point[1]][point[2]].typeID == 1) {                
                    positions.emplace_back(dx, dy, dz);
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

    // std::cout << "Eigen::Vector3d normal: " << normal << std::endl;
    // 步骤5: 写入法线数据
    for (size_t i = 0; i < 3; ++i) {
        Cells[posInt[0]][posInt[1]][posInt[2]].normal[i] = normal[i];
    }
}


void World::update_normal_in_matrix() {
    std::vector<int3> unique_points;

    size_t update_film_etch_size = update_film_etch.size();
    for (int i = 0; i < update_film_etch_size; ++i) {
        int3 posInt = update_film_etch[i];

        for (int dx = -3; dx <= 3; ++dx) {
            for (int dy = -3; dy <= 3; ++dy) {
                for (int dz = -3; dz <= 3; ++dz) {
                    int xi = posInt[0] + dx;
                    int yi = posInt[1] + dy;
                    int zi = posInt[2] + dz;
                    int3 point = {xi, yi, zi};
                    point = mirror_index(point);

                    if (Cells[point[0]][point[1]][point[2]].typeID == 1) {
                        // int3 point = {xi, yi, zi};
                        unique_points.push_back(point);
                    }
                    
                }
            }
        }
    }

    size_t unique_points_size = unique_points.size();

    // std::cout << "unique_points: ";
    // for (size_t f = 0; f < unique_points.size(); ++f) {
    //     std::cout << unique_points[f] << '\n';
    // }
    // std::cout << '\n';

    for (int j = 0; j < unique_points_size; ++j) {
        get_normal_from_grid(unique_points[j]);
    }
}

void World::print_Cells(){
        // 将数据复制到 NumPy 数组

    int surface = 0;
    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                if(Cells[i][j][k].typeID == 1) {
                    surface++;
                    std::cout << "surface: " << i << " " << j << " " << k << " " << Cells[i][j][k].normal << std::endl;
                }
            }
        }
    }
    std::cout << "surfacecount: "<< surface << std::endl;
}

// void get_normal_from_grid_Cell(
//     py::array_t<Cell, py::array::c_style> cell_array,
//     // py::array_t<int> film,
//     // py::array_t<double> normal_matrix,
//     // int mirrorGap,
//     py::array_t<int> point
// ) {
//     // 获取原始指针和维度信息
//     auto cell_buf = cell_array.request();
//     auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

//     // auto film_buf = film.request();
//     // int* film_ptr = static_cast<int*>(film_buf.ptr);
//     const int dim_x = cell_array.shape(0);
//     const int dim_y = cell_array.shape(0);
//     const int dim_z = cell_array.shape(0);

//     // auto normal_buf = normal_matrix.request();
//     // double* normal_ptr = static_cast<double*>(normal_buf.ptr);

//     auto point_buf = point.request();
//     int* point_ptr = static_cast<int*>(point_buf.ptr);
//     int x = point_ptr[0];
//     int y = point_ptr[1];
//     int z = point_ptr[2];

//     // 三维索引计算函数
//     auto film_index = [dim_y, dim_z](int x, int y, int z) {
//         return x * dim_y * dim_z + y * dim_z + z;
//     };

//     auto normal_index = [dim_y, dim_z](int x, int y, int z) {
//         return x * dim_y * dim_z + y * dim_z + z;
//     };

//     // 步骤1: 收集7x7x7立方体内值为1的坐标
//     std::vector<Eigen::Vector3d> positions;
//     for (int dx = -3; dx <= 3; ++dx) {
//         for (int dy = -3; dy <= 3; ++dy) {
//             for (int dz = -3; dz <= 3; ++dz) {
//                 int xi = x + dx;
//                 int yi = y + dy;
//                 int zi = z + dz;
                
//                 if (xi >= 0 && xi < dim_x && 
//                     yi >= 0 && yi < dim_y && 
//                     zi >= 0 && zi < dim_z) 
//                 {
//                     // 直接通过内存指针访问
//                     if (cell_ptr[film_index(xi, yi, zi)].id == 1) {
//                         positions.emplace_back(dx, dy, dz);
//                     }
//                 }
//             }
//         }
//     }

//     // ... [保持原有计算逻辑不变]
//     // 如果没有有效点，返回原矩阵
//     if (positions.empty()) {
//         std::cout << "svd矩阵为空" << std::endl;
//     }

//     // 步骤2: 计算均值
//     Eigen::Vector3d mean(0, 0, 0);
//     for (const auto& pos : positions) {
//         mean += pos;
//     }
//     mean /= positions.size();

//     // 步骤3: 计算协方差矩阵
//     Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
//     for (const auto& pos : positions) {
//         Eigen::Vector3d centered = pos - mean;
//         cov += centered * centered.transpose();
//     }

//     // 步骤4: SVD分解
//     Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
//     Eigen::Vector3d normal = svd.matrixU().col(2); // 最小特征值对应的特征向量

//     // 步骤5: 写入法线数据
//     if (x >= 0 && x < dim_x && 
//         y >= 0 && y < dim_y && 
//         z >= 0 && z < dim_z) 
//     {
//         const size_t base_idx = normal_index(x, y, z);
//         if (z + 2 < dim_z) {
//             cell_ptr[base_idx].normal[0] = normal[0];
//             cell_ptr[base_idx].normal[1] = normal[1];
//             cell_ptr[base_idx].normal[2] = normal[2];
//         } else {
//             // 处理边界情况
//             const int available = dim_z - z;
//             for (int i = 0; i < available; ++i) {
//                 cell_ptr[base_idx].normal[i] = normal[i];
//             }
//         }
//     }
// }


// void get_normal_from_grid_Cell_toc(
//     py::array_t<Cell, py::array::c_style> cell_array,
//     // py::array_t<int> film,
//     // py::array_t<double> normal_matrix,
//     // int mirrorGap,
//     // std:vector<int> point
//     int x,
//     int y,
//     int z
// ) {
//     // 获取原始指针和维度信息
//     auto cell_buf = cell_array.request();
//     auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

//     // auto film_buf = film.request();
//     // int* film_ptr = static_cast<int*>(film_buf.ptr);
//     const int dim_x = cell_array.shape(0);
//     const int dim_y = cell_array.shape(0);
//     const int dim_z = cell_array.shape(0);

//     // auto normal_buf = normal_matrix.request();
//     // double* normal_ptr = static_cast<double*>(normal_buf.ptr);

//     // auto point_buf = point.request();
//     // int* point_ptr = static_cast<int*>(point_buf.ptr);
//     // int x = point_ptr[0];
//     // int y = point_ptr[1];
//     // int z = point_ptr[2];

//     // 三维索引计算函数
//     auto film_index = [dim_y, dim_z](int x, int y, int z) {
//         return x * dim_y * dim_z + y * dim_z + z;
//     };

//     auto normal_index = [dim_y, dim_z](int x, int y, int z) {
//         return x * dim_y * dim_z + y * dim_z + z;
//     };

//     // 步骤1: 收集7x7x7立方体内值为1的坐标
//     std::vector<Eigen::Vector3d> positions;
//     for (int dx = -3; dx <= 3; ++dx) {
//         for (int dy = -3; dy <= 3; ++dy) {
//             for (int dz = -3; dz <= 3; ++dz) {
//                 int xi = x + dx;
//                 int yi = y + dy;
//                 int zi = z + dz;
                
//                 if (xi >= 0 && xi < dim_x && 
//                     yi >= 0 && yi < dim_y && 
//                     zi >= 0 && zi < dim_z) 
//                 {
//                     // 直接通过内存指针访问
//                     if (cell_ptr[film_index(xi, yi, zi)].id == 1) {
//                         positions.emplace_back(dx, dy, dz);
//                     }
//                 }
//             }
//         }
//     }

//     // ... [保持原有计算逻辑不变]
//     // 如果没有有效点，返回原矩阵
//     if (positions.empty()) {
//         std::cout << "svd矩阵为空" << std::endl;
//     }

//     // 步骤2: 计算均值
//     Eigen::Vector3d mean(0, 0, 0);
//     for (const auto& pos : positions) {
//         mean += pos;
//     }
//     mean /= positions.size();

//     // 步骤3: 计算协方差矩阵
//     Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
//     for (const auto& pos : positions) {
//         Eigen::Vector3d centered = pos - mean;
//         cov += centered * centered.transpose();
//     }

//     // 步骤4: SVD分解
//     Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
//     Eigen::Vector3d normal = svd.matrixU().col(2); // 最小特征值对应的特征向量

//     // 步骤5: 写入法线数据
//     if (x >= 0 && x < dim_x && 
//         y >= 0 && y < dim_y && 
//         z >= 0 && z < dim_z) 
//     {
//         const size_t base_idx = normal_index(x, y, z);
//         if (z + 2 < dim_z) {
//             cell_ptr[base_idx].normal[0] = normal[0];
//             cell_ptr[base_idx].normal[1] = normal[1];
//             cell_ptr[base_idx].normal[2] = normal[2];
//         } else {
//             // 处理边界情况
//             const int available = dim_z - z;
//             for (int i = 0; i < available; ++i) {
//                 cell_ptr[base_idx].normal[i] = normal[i];
//             }
//         }
//     }
// }


// void update_normal_in_matrix(
//     py::array_t<Cell, py::array::c_style> cell_array,
//     py::array_t<int, py::array::c_style> point_to_change
// ) {
//     // 获取输入数组信息
//     auto cell_buf = cell_array.request();
//     auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

//     const int dim_x = cell_array.shape(0);
//     const int dim_y = cell_array.shape(0);
//     const int dim_z = cell_array.shape(0);

//     // auto film_mirror_buf = film_label_mirror.request();
//     // auto* film_mirror = static_cast<LabeledNormal*>(film_mirror_buf.ptr);
//     // const int dim_x = film_mirror_buf.shape[0];
//     // const int dim_y = film_mirror_buf.shape[1];
//     // const int dim_z = film_mirror_buf.shape[2];

//     // auto film_normal_buf = film_label_normal.request();
//     // auto* film_normal = static_cast<double*>(film_normal_buf.ptr);

//     auto points_buf = point_to_change.request();
//     auto* points = static_cast<int*>(points_buf.ptr);
//     const int num_points = point_to_change.shape(0);

//     // 三维索引计算函数
//     auto film_index = [dim_y, dim_z](int x, int y, int z) {
//         return x * dim_y * dim_z + y * dim_z + z;
//     };

//     // 步骤1: 收集所有需要处理的点
//     // std::unordered_set<std::tuple<int, int, int>> unique_points;
//     std::vector<Eigen::Vector3d> unique_points;
//     // #pragma omp parallel for
//     for (int i = 0; i < num_points; ++i) {
//         int x = points[i * 3];
//         int y = points[i * 3 + 1];
//         int z = points[i * 3 + 2];

//         // 边界检查
//         int x_start = std::max(x - 3, 0);
//         int x_end = std::min(x + 4, dim_x);
//         int y_start = std::max(y - 3, 0);
//         int y_end = std::min(y + 4, dim_y);
//         int z_start = std::max(z - 3, 0);
//         int z_end = std::min(z + 4, dim_z);

//         // 遍历7x7x7立方体
//         for (int xi = x_start; xi < x_end; ++xi) {
//             for (int yi = y_start; yi < y_end; ++yi) {
//                 for (int zi = z_start; zi < z_end; ++zi) {
//                     if (cell_ptr[film_index(xi, yi, zi)].id == 1) {
//                         // #pragma omp critical
//                         unique_points.emplace_back(xi,yi,zi);
//                     }
//                 }
//             }
//         }
//     }

//     #pragma omp parallel for
//     for (const auto& pos : unique_points) {
//         // 获取唯一点坐标
//         int x = pos[0];
//         int y = pos[1];
//         int z = pos[2];

//         // 调用优化后的法线计算函数
//         get_normal_from_grid_Cell_toc(
//             cell_array,
//             x, y, z
//         );
//     }
//     // 步骤2: 去重后更新法线
//     // #pragma omp parallel for
//     // for(const auto& pos : unique_points) {
//     //     // 调用优化后的法线计算函数
//     //     get_normal_from_grid_Cell_toc(
//     //     cell_array,
//     //     pos
//     //     );
//     // }

//     // return film_label_normal;
// }


// void inputCell(
//     py::array_t<Cell, py::array::c_style> cell,
// ) {
//     // 获取输入数组信息
//     auto cell_buf = cell.request();
//     auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

//     const int dim_x = cell.shape(0);
//     const int dim_y = cell.shape(0);
//     const int dim_z = cell.shape(0);
//     std::cout << "inputCell:" << dim_x << '_' << dim_y  << '_' << dim_z << std::endl;
// }


