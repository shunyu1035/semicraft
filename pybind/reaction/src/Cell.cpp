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


// void World::film_add(int3 posInt, std::vector<int> react_add){
//     for (int a=0; a<FILMSIZE; ++a){
//         // std::cout << react_add[a] <<  std::endl;
//         Cells[posInt[0]][posInt[1]][posInt[2]].film[a] += react_add[a];
//         // std::cout << Cells[posInt[0]][posInt[1]][posInt[2]].film[a] <<  std::endl;
//     }
// }

void World::film_add(int3 posInt, const std::vector<int>& react_add) {
    Cell& cell = Cells[posInt[0]][posInt[1]][posInt[2]];
    std::lock_guard<std::mutex> lock(cell.film_mutex); // 加锁

    for (int a = 0; a < FILMSIZE; ++a) {
        cell.film[a] += react_add[a];
    }
}

// void World::film_add(int3 posInt, const std::vector<int>& react_add) {
//     // 假设 Cells 是一个三维容器：std::vector<std::vector<std::vector<Cell>>>
//     Cell& cell = Cells[posInt[0]][posInt[1]][posInt[2]];
    
//     // 对 film 数组中的每个元素使用原子操作进行加法
//     for (size_t a = 0; a < cell.film.size(); ++a) {
//         cell.film[a].fetch_add(react_add[a], std::memory_order_relaxed);
//     }
// }

double World::react_prob_chemical_angle(double angle_rad) {
    if (angle_rad < chemical_angle_v1) {
        return 0.9999;
    } else if ((angle_rad >= chemical_angle_v1) && (angle_rad < chemical_angle_v2)) {
        return (angle_rad - chemical_angle_v2)/(chemical_angle_v1 - chemical_angle_v2);
    } else if (angle_rad >= chemical_angle_v2) {
        return 0.0;
    }
    return 0;
}

std::vector<int> World::sticking_probability_structed(const Particle particle, const Cell cell, double angle_rad, Rnd &rnd) {

    std::vector<int> sticking_acceptList(FILMSIZE, 0);
    std::vector<double> choice(FILMSIZE);

    // 使用外部传入的 rnd 生成 [0,1) 内的随机数
    for (int i = 0; i < FILMSIZE; ++i) {
        choice[i] = rnd();
    }

    int energy_range = 0;
    double sticking_rate = 0.0;
    int particle_id = particle.id;
    // int particle_id = (particle.id > ArgonID) ? ArgonID : particle.id;

    for (int j = 0; j < FILMSIZE; ++j) {
        if (cell.film[j] <= 0) {
            choice[j] = 10000.0; // choice can be efficient large for no reaction
        }

        if (particle_id == ArgonID) {
            energy_range = 0;
            for (int e = 0; e < (int)rn_energy.size(); ++e) {
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
            sticking_rate *= react_prob_chemical_angle(angle_rad);
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

void World::update_Cells_inthread(int3 posInt){
    std::vector<int3> local_point_nn;
    std::vector<int3> local_point_nn_under;
    std::vector<int3> local_point_nn_vaccum;

    // for (size_t i=0; i<update_film_etch_size; i++){
    //     int3 posInt = update_film_etch[i];

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

        }
    }

    // update_normal_in_matrix();
    update_normal_in_matrix_inthread(posInt);

    // update_film_etch.resize(0);
    // } 
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

void World::update_normal_in_matrix_inthread(int3 posInt) {
    std::vector<int3> unique_points;

    // size_t update_film_etch_size = update_film_etch.size();
    // for (int i = 0; i < update_film_etch_size; ++i) {
    // int3 posInt = update_film_etch[i];

    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dz = -3; dz <= 3; ++dz) {
                int xi = posInt[0] + dx;
                int yi = posInt[1] + dy;
                int zi = posInt[2] + dz;
                int3 point = {xi, yi, zi};
                point = mirror_index(point);

                if (Cells[point[0]][point[1]][point[2]].typeID == 1 || Cells[point[0]][point[1]][point[2]].typeID == 2) {
                    // int3 point = {xi, yi, zi};
                    unique_points.push_back(point);
                }
                
            }
        }
        // }
    }

    int unique_points_size = unique_points.size();

    for (int j = 0; j < unique_points_size; ++j) {
        get_normal_from_grid(unique_points[j]);
    }
}



void World::update_normal_in_matrix() {
    std::vector<int3> unique_points;

    size_t update_film_etch_size = update_film_etch.size();
    for (size_t i = 0; i < update_film_etch_size; ++i) {
        int3 posInt = update_film_etch[i];

        for (int dx = -3; dx <= 3; ++dx) {
            for (int dy = -3; dy <= 3; ++dy) {
                for (int dz = -3; dz <= 3; ++dz) {
                    int xi = posInt[0] + dx;
                    int yi = posInt[1] + dy;
                    int zi = posInt[2] + dz;
                    int3 point = {xi, yi, zi};
                    point = mirror_index(point);

                    if (Cells[point[0]][point[1]][point[2]].typeID == 1 || Cells[point[0]][point[1]][point[2]].typeID == 2) {
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

    for (size_t j = 0; j < unique_points_size; ++j) {
        get_normal_from_grid(unique_points[j]);
    }
}

void World::print_Cells(){
        // 将数据复制到 NumPy 数组

    int surface = 0;
    for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
            for (int k = 0; k < nk; ++k) {
                if(Cells[i][j][k].typeID == 1) {
                    surface++;
                    std::cout << "surface: " << i << " " << j << " " << k << " " << Cells[i][j][k].normal << std::endl;
                }
            }
        }
    }
    std::cout << "surfacecount: "<< surface << std::endl;
}

