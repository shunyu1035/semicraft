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


int3 World::find_depo_cell(int3 posInt, Rnd &rnd){
    std::vector<int3> local_point_nn;
    std::vector<double> label_to_depo;
    int3 depo_cell;
    bool all_full = true;
    // local_point_nn.resize(6);

    if (Cells[posInt[0]][posInt[1]][posInt[2]].typeID == 1){
        local_point_nn.resize(6);
        label_to_depo.resize(6);
        for (size_t j=0; j<6; ++j){
            local_point_nn[j] = mirror_index(posInt + grid_cross[j]);

            if (Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID == -1){
                label_to_depo[j] = rnd();
            }
        }

        int depo_choice = find_max_position(label_to_depo);
        depo_cell = mirror_index(posInt + grid_cross[depo_choice]);

        return depo_cell;

    }
    else {
        local_point_nn.resize(6);
        label_to_depo.resize(6);
        for (size_t j=0; j<6; ++j){
            local_point_nn[j] = mirror_index(posInt + grid_cross[j]);

            if (film_full(local_point_nn[j]) == false){
                // all_full = true;
                all_full = false;
                label_to_depo[j] = rnd();
            }
        }

        if (all_full){
            local_point_nn.resize(20);
            label_to_depo.resize(20);
            for (size_t j=0; j<20; ++j){
                local_point_nn[j] = mirror_index(posInt + grid_cube[j]);
                if (Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID == -1){
                    label_to_depo[j] = rnd();
                }
            }
            int depo_choice = find_max_position(label_to_depo);
            depo_cell = mirror_index(posInt + grid_cube[depo_choice]);

        }
        else{
            int depo_choice = find_max_position(label_to_depo);
            depo_cell = mirror_index(posInt + grid_cube[depo_choice]);
        }

    }

    return depo_cell;
}


int3 World::surface_diffusion(int3 posInt, Rnd &rnd){
    std::vector<int3> local_point_nn;
    std::vector<double> potential_nn(26, -100);
    int3 diffusion_cell;

    local_point_nn.resize(26);
    double origin_h = Cells[posInt[0]][posInt[1]][posInt[2]].potential;
    for (size_t j=0; j<26; ++j){
        local_point_nn[j] = mirror_index(posInt + grid_cube_all[j]);
        if (Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID == 1){
            potential_nn[j] = origin_h - Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].potential;
        }
    }

    int diffusion_choice = find_max_position(potential_nn);
    double potential_gap = potential_nn[diffusion_choice];
    diffusion_cell = mirror_index(posInt + grid_cube_all[diffusion_choice]);

    if (potential_gap * diffusion_coeffient > rnd()) {
        return diffusion_cell;
    }

    return posInt;    
}


void World::update_Cells_inthread_depo(int3 posInt){
    std::vector<int3> local_point_nn;
    // std::vector<int3> local_point_nn_under;
    std::vector<int3> local_point_nn_vaccum;

    // for (size_t i=0; i<update_film_etch_size; i++){
    //     int3 posInt = update_film_etch[i];

        // std::cout << "posInt: " << posInt << std::endl;

    local_point_nn.resize(6);
    Cells[posInt[0]][posInt[1]][posInt[2]].typeID = 1;
    for (size_t j=0; j<6; ++j){
        local_point_nn[j] = mirror_index(posInt + grid_cross[j]);
        

        // std::cout << "inloop local_point_nn: " << local_point_nn[j] << std::endl;

        if (Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID == 0){
            Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = -1;

            // local_point_nn_under.resize(6);
            // for (size_t k=0; k<6; ++k){
            //     local_point_nn_under[k] = mirror_index(local_point_nn[j] + grid_cross[k]);
            //     if (Cells[local_point_nn_under[k][0]][local_point_nn_under[k][1]][local_point_nn_under[k][2]].typeID == 3){
            //         Cells[local_point_nn_under[k][0]][local_point_nn_under[k][1]][local_point_nn_under[k][2]].typeID = 2;
            //     }
            // }
            // std::cout << "local_point_nn_vaccum: ";
            // for (size_t q = 0; q < local_point_nn_under.size(); ++q) {
            //     std::cout << local_point_nn_under[q] << '\n';
            // }
            // std::cout << '\n';
        }
        else if (Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID == 1) {

            // 先把-1的label变成0
            Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 2;

            local_point_nn_vaccum.resize(6);
            for (size_t l=0; l<6; ++l){
                local_point_nn_vaccum[l] = mirror_index(local_point_nn[j] + grid_cross[l]);

                // std::cout << "local_point_nn_vaccum: " << local_point_nn_vaccum[l] << std::endl;
                // Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 0;
                if (Cells[local_point_nn_vaccum[l][0]][local_point_nn_vaccum[l][1]][local_point_nn_vaccum[l][2]].typeID == -1){
                    // 再把相邻为1的label变为-1
                    Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 1;

                    break;
                }
            }

        }
    }

    // update_normal_in_matrix();
    update_normal_in_matrix_inthread(posInt);

    // update_film_etch.resize(0);
    // } 
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

            // 先把-1的label变成0
            Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 0;

            local_point_nn_vaccum.resize(6);
            for (size_t l=0; l<6; ++l){
                local_point_nn_vaccum[l] = mirror_index(local_point_nn[j] + grid_cross[l]);

                // std::cout << "local_point_nn_vaccum: " << local_point_nn_vaccum[l] << std::endl;
                // Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 0;
                if (Cells[local_point_nn_vaccum[l][0]][local_point_nn_vaccum[l][1]][local_point_nn_vaccum[l][2]].typeID == 1){
                    // 再把相邻为1的label变为-1
                    Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = -1;
                    
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

                    Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = 0;

                    local_point_nn_vaccum.resize(6);
                    for (size_t l=0; l<6; ++l){
                        local_point_nn_vaccum[l] = mirror_index(local_point_nn[j] + grid_cross[l]);

                        // std::cout << "local_point_nn_vaccum: " << local_point_nn_vaccum[l] << std::endl;

                        if (Cells[local_point_nn_vaccum[l][0]][local_point_nn_vaccum[l][1]][local_point_nn_vaccum[l][2]].typeID == 1){
                            Cells[local_point_nn[j][0]][local_point_nn[j][1]][local_point_nn[j][2]].typeID = -1;
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

    // 判断法线朝向
    double3 direction{0, 0, 1};
    // double dot_direction_normal;

    int top = Cells[posInt[0]][posInt[1]][mirror_z(posInt[2]+3)].typeID;
    int bottom = Cells[posInt[0]][posInt[1]][mirror_z(posInt[2]-3)].typeID;
    int front = Cells[mirror_x(posInt[0]-3)][posInt[1]][posInt[2]].typeID;
    int back = Cells[mirror_x(posInt[0]+3)][posInt[1]][posInt[2]].typeID;
    int left = Cells[posInt[0]][mirror_y(posInt[1]-3)][posInt[2]].typeID;
    int right = Cells[posInt[0]][mirror_y(posInt[1]+3)][posInt[2]].typeID;

    if ((top == 3 || top == 2) && (bottom == 0)) {
        direction = {0, 0, -1};
    }
    else if ((front == 3 || front == 2) && (back == 0)) {
        direction = {1, 0, 0};
    }
    else if ((back == 3 || back == 2) && (front == 0)) {
        direction = {-1, 0, 0};
    }
    else if ((left == 3 || left == 2) && (right == 0)) {
        direction = {0, 1, 0};
    }
    else if ((right == 3 || right == 2) && (left == 0)) {
        direction = {0, -1, 0};
    }


    // 如果没有有效点，返回原矩阵
    if (positions.empty()) {
        std::cout << "svd矩阵为空" << std::endl;
        std::cout << "posInt: " << posInt << std::endl;
    }

    // 步骤2: 计算均值
    Eigen::Vector3d mean(0, 0, 0);
    for (const auto& pos : positions) {
        mean += pos;
    }
    mean /= positions.size();

    // 判断凹凸朝向
    // double3 posInt_d3{(double)posInt[0], (double)posInt[1], (double)posInt[2]};
    double3 origin_pt{0, 0, 0};
    double3 mean3{mean[0], mean[1], mean[2]};
    double3 curvature_vector = origin_pt - mean3;

    // 步骤3: 计算协方差矩阵
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pos : positions) {
        Eigen::Vector3d centered = pos - mean;
        cov += centered * centered.transpose();
    }

    // 步骤4: SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
    Eigen::Vector3d normal = svd.matrixU().col(2); // 最小特征值对应的特征向量

    // 计算曲率
    Eigen::Vector3d S = svd.singularValues();  // s0 ≥ s1 ≥ s2
    double sum = S.sum();
    double lambda_min = S(2);
    double curvature = 0;
    if (sum > 0) {
        curvature = lambda_min / sum;
    }


    // 判断法线朝向
    double3 normal3{normal[0], normal[1], normal[2]};
    double dot_direction_normal = dot(normal3, direction);

    double sign_dot = 1;
    if (dot_direction_normal < 0 ){
        sign_dot = -1;
    }
    double3 normal_corrected = normal3 * sign_dot;

    // 判断凹凸朝向， 曲率正负
    double curvature_vector_dot = dot(curvature_vector, normal_corrected);
    double curvature_sign_dot = 1;
    if (curvature_vector_dot < 0 ){
        curvature_sign_dot = -1;
    }

    // 步骤5: 写入法线数据
    Cells[posInt[0]][posInt[1]][posInt[2]].normal = normal_corrected;
    Cells[posInt[0]][posInt[1]][posInt[2]].potential = curvature_sign_dot * curvature;

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

