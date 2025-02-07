#include "Particle.h"
#include "Field.h"
#include <pybind11/iostream.h>  // 用于重定向输出
#include <iostream>
#include <pybind11/eigen.h>  // 支持 Eigen 与 NumPy 互操作
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <algorithm>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <omp.h>
#include <pybind11/pybind11.h>
// #include "Cell.h"

namespace py = pybind11;


// 绑定vec3类型
template<typename T>
void bind_vec3(py::module &m, const std::string &typestr) {
    py::class_<vec3<T>>(m, typestr.c_str())
        .def(py::init<>())
        .def(py::init<T, T, T>())
        .def("__getitem__", [](const vec3<T> &v, int i) {
            if (i < 0 || i >= 3) throw py::index_error();
            return v(i);
        })
        .def("__setitem__", [](vec3<T> &v, int i, T val) {
            if (i < 0 || i >= 3) throw py::index_error();
            v[i] = val;
        })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * T())
        .def(T() * py::self)
        .def("__repr__", [](const vec3<T> &v) {
            return "vec3(" + std::to_string(v(0)) + ", " 
                         + std::to_string(v(1)) + ", "
                         + std::to_string(v(2)) + ")";
        });
}

// 粒子结构体绑定
void bind_particle(py::module &m) {
    py::class_<Particle>(m, "Particle")
        .def(py::init<double3, double3, double, int>())
        .def_readwrite("pos", &Particle::pos)
        .def_readwrite("vel", &Particle::vel)
        .def_readwrite("E", &Particle::E)
        .def_readwrite("id", &Particle::id)
        .def("__repr__", [](const Particle &p) {
            return "Particle(id=" + std::to_string(p.id) 
                 + ", E=" + std::to_string(p.E) + ")";
        });
}

// Cell结构体绑定
void bind_Cell(py::module &m) {
    py::class_<Cell>(m, "Cell")
        .def(py::init<int, std::array<int, 3>, std::array<int, 5>, std::array<double, 3>>())
        .def_readwrite("id", &Cell::id)
        .def_readwrite("index", &Cell::index)
        .def_readwrite("film", &Cell::film)
        .def_readwrite("normal", &Cell::normal)
        .def("__repr__", [](const Cell &p) {
            return "Cell(id=" + std::to_string(p.id) + ")";
        });
}



class Simulation {
private:
    std::vector<Particle> particles;
    std::mt19937 rng;  // 随机数引擎

    std::vector<std::vector<std::vector<Cell>>> Cells;
    // 生成麦克斯韦分布速度
    double3 maxwell_velocity(double T) {
        std::normal_distribution<double> dist(0.0, sqrt(T));
        return {dist(rng), dist(rng), dist(rng)};
    }
    World world;  // 注意：World 必须有合适的构造函数

    std::vector<std::vector<std::vector<int>>> react_table_equation;
    std::vector<std::vector<int>> react_type_table;
    std::vector<double> react_prob_chemical;
    std::vector<double> react_yield_p0;
    std::vector<std::vector<double>> rn_coeffcients;

    // std::unique_ptr<World> world_ptr;

public:
    // 构造函数：初始化随机数引擎
    // Simulation(int seed = 42) : rng(seed) {}
    // Simulation(int seed, int ni, int nj, int nk) : rng(seed){}
    Simulation(int seed, int ni, int nj, int nk) : rng(seed), world(ni, nj, nk) {}
    // World world(10, 10, 10)


    // 从 NumPy 数组设置数据，要求输入必须为三维数组
    void set_react_table_equation(py::array_t<int> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 3) {
            throw std::runtime_error("输入数组必须是三维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        ssize_t dim1 = buf.shape[1];
        ssize_t dim2 = buf.shape[2];
        int* ptr = static_cast<int*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        react_table_equation.resize(dim0);
        for (ssize_t i = 0; i < dim0; ++i) {
            react_table_equation[i].resize(dim1);
            for (ssize_t j = 0; j < dim1; ++j) {
                react_table_equation[i][j].resize(dim2);
                // 计算内存中对应的起始偏移：
                // 对于形状为 (dim0, dim1, dim2) 的数组，元素索引 (i, j, k) 的偏移量为: i * (dim1*dim2) + j * dim2 + k
                ssize_t offset = i * (dim1 * dim2) + j * dim2;
                for (ssize_t k = 0; k < dim2; ++k) {
                    react_table_equation[i][j][k] = ptr[offset + k];
                }
            }
        }
    }

    // 从 NumPy 数组设置数据，要求输入必须为三维数组
    void set_react_type_table(py::array_t<int> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("react_type_table 输入数组必须是2维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        ssize_t dim1 = buf.shape[1];
        int* ptr = static_cast<int*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        react_type_table.resize(dim0);
        for (ssize_t i = 0; i < dim0; ++i) {
            react_type_table[i].resize(dim1);
            for (ssize_t j = 0; j < dim1; ++j) {
                // 计算内存中对应的起始偏移：
                ssize_t offset = i * dim1 + j;
                react_type_table[i][j] = ptr[offset];

            }
        }
    }


    // 从 NumPy 数组设置数据，要求输入必须为三维数组
    void set_rn_coeffcients(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("rn_coeffcients 输入数组必须是2维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        ssize_t dim1 = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        rn_coeffcients.resize(dim0);
        for (ssize_t i = 0; i < dim0; ++i) {
            rn_coeffcients[i].resize(dim1);
            for (ssize_t j = 0; j < dim1; ++j) {
                // 计算内存中对应的起始偏移：
                ssize_t offset = i * dim1 + j;
                rn_coeffcients[i][j] = ptr[offset];

            }
        }
    }


    // 从 NumPy 数组设置数据，要求输入必须为三维数组
    void set_react_yield_p0(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("react_yield_p0 输入数组必须是1维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        double* ptr = static_cast<double*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        react_yield_p0.resize(dim0);
        for (ssize_t i = 0; i < dim0; ++i) {
            // 计算内存中对应的起始偏移：
            ssize_t offset = i;
            react_yield_p0[i] = ptr[offset];
        }
    }


    // 从 NumPy 数组设置数据，要求输入必须为三维数组
    void set_react_prob_chemical(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("react_prob_chemical 输入数组必须是2维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        double* ptr = static_cast<double*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        react_prob_chemical.resize(dim0);
        for (ssize_t i = 0; i < dim0; ++i) {
            // 计算内存中对应的起始偏移：
            ssize_t offset = i;
            react_prob_chemical[i] = ptr[offset];
        }
    }



    void set_all_parameters(py::array_t<int> react_table_equation, 
                            py::array_t<int> react_type_table,
                            py::array_t<double> react_prob_chemical,
                            py::array_t<double> react_yield_p0,
                            py::array_t<double> rn_coeffcients){

        set_react_table_equation(react_table_equation);
        set_react_type_table(react_type_table);
        set_react_prob_chemical(react_prob_chemical);
        set_react_yield_p0(react_yield_p0);
        set_rn_coeffcients(rn_coeffcients);
    }


    // 打印内部数据
    void print_react_table_equation() const {
        if (react_table_equation.empty()) return;
        ssize_t dim0 = react_table_equation.size();
        ssize_t dim1 = react_table_equation[0].size();
        ssize_t dim2 = react_table_equation[0][0].size();
        for (ssize_t i = 0; i < dim0; ++i) {
            std::cout << "react_table_equation " << i << ":" << std::endl;
            for (ssize_t j = 0; j < dim1; ++j) {
                for (ssize_t k = 0; k < dim2; ++k) {
                    std::cout << react_table_equation[i][j][k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }



    void runSimulation(){
        double3 xm = world.getXm();
        std::cout << "World xm: " << xm << std::endl; 

        world.set_cell(Cells);
        Species sp("test", 1, world);
    }

    void testWorld(){
        // std::cout << "World xm: "  << std::endl;
        double3 xm = world.getXm();
        std::cout << "World xm: " << xm << std::endl;
    }

    void inputCell(
        py::array_t<Cell, py::array::c_style> cell
    ) {
        // 获取输入数组信息
        auto cell_buf = cell.request();
        auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

        size_t dim_x = cell.shape(0);
        size_t dim_y = cell.shape(1);
        size_t dim_z = cell.shape(2);

        // World world(dim_x, dim_x, dim_x);
        // world_ptr = std::make_unique<World>(dim_x, dim_y, dim_z);
        std::cout << "inputCell:" << dim_x << '_' << dim_y  << '_' << dim_z << std::endl;

        Cells.resize(dim_x);
        for (size_t i = 0; i < dim_x; ++i) {
            Cells[i].resize(dim_y);
            for (size_t j = 0; j < dim_y; ++j) {
                size_t offset = (i * dim_y + j) * dim_z;
                Cells[i][j].assign(cell_ptr + offset, cell_ptr + offset + dim_z);
                // Cells[i][j].assign(cell_ptr + i * j * dim_z, cell_ptr + i * (j + 1) * dim_z);
            }
        }
    }

//    // 从 NumPy 数组设置数据，要求输入必须为三维数组
//     void set_react_prob_chemical(py::array_t<double> arr) {
//         py::buffer_info buf = arr.request();
//         if (buf.ndim != 1) {
//             throw std::runtime_error("react_prob_chemical 输入数组必须是2维的");
//         }
//         // 获取每一维的大小
//         ssize_t dim0 = buf.shape[0];
//         double* ptr = static_cast<double*>(buf.ptr);

//         // 按三维结构分配 react_table_equation
//         react_prob_chemical.resize(dim0);
//         for (ssize_t i = 0; i < dim0; ++i) {
//             // 计算内存中对应的起始偏移：
//             ssize_t offset = i;
//             react_prob_chemical[i] = ptr[offset];
//         }
//     }


    void inputParticle(
        py::array_t<double> pos_py,
        py::array_t<double> vel_py,
        py::array_t<double> E_py,
        py::array_t<int> id_py
    ) {
        // 获取输入数组信息
        py::buffer_info  pos_py_buf = pos_py.request();
        py::buffer_info  vel_py_buf = vel_py.request();
        py::buffer_info  E_py_buf = E_py.request();
        py::buffer_info  id_py_buf = id_py.request();

        if (pos_py_buf.ndim < 1) {
            throw std::runtime_error("pos_py 数组至少应该是一维");
        }

        double* pos_ptr = static_cast<double*>(pos_py_buf.ptr);
        double* vel_ptr = static_cast<double*>(vel_py_buf.ptr);
        double* E_ptr = static_cast<double*>(E_py_buf.ptr);
        int* id_ptr = static_cast<int*>(id_py_buf.ptr);

        size_t posN = pos_py_buf.shape[0];
        size_t velN = vel_py_buf.shape[0];
        size_t EN = E_py_buf.shape[0];
        size_t idN = id_py_buf.shape[0];
        if (((posN != velN && posN != EN) && posN != idN)) {
            throw std::runtime_error("particle 维度不一致");
        }
        std::cout << "inputParticle: " << posN << std::endl;

        particles.reserve(posN);
        for(size_t i=0; i<posN; ++i){
            ssize_t offset3 = i * 3;
            double3 pos = {pos_ptr[offset3], pos_ptr[offset3 + 1], pos_ptr[offset3 + 2]};
            double3 vel = {vel_ptr[offset3], vel_ptr[offset3 + 1], vel_ptr[offset3 + 2]};
            particles.emplace_back(pos, vel, E_ptr[i], id_ptr[i]);
        }

        std::cout << "ParticleSize: " << particles.size() << std::endl;
        // for(size_t i=0; i<num; ++i){
        //     particles[i] = particle_ptr[i];
        // }
    }

    // 获取所有粒子 (Python 访问接口)
    const std::vector<std::vector<std::vector<Cell>>>& getCells() const {
        return Cells;
    }


    // 初始化粒子群
    void initialize(int num_particles, double T, double E, int id , double box_size = 100.0) {
        std::uniform_real_distribution<double> pos_dist(0.0, box_size);
        particles.reserve(num_particles);

        for(int i=0; i<num_particles; ++i){
            double3 pos{pos_dist(rng), pos_dist(rng), pos_dist(rng)};
            double3 vel = maxwell_velocity(T);
            particles.emplace_back(pos, vel, E, id);
        }
    }

    // 添加粒子（Python 交互接口）
    void addParticle(double3 pos, double3 vel, double E, int id) {
        particles.emplace_back(pos, vel, E, id);
    }



    // move
    void moveParticle(int id) {
        particles[id].pos += particles[id].vel;
    }

    // print
    void printParticle(int id) {
        std::cout << "particles["<< id <<"].pos: " << particles[id].pos << std::endl;    // 输出: particles[id].pos
        std::cout << "particles["<< id <<"].vel: " << particles[id].vel << std::endl;    // 输出: particles[id].pos
    }

    // cross test
    void crossTest(int i, int j) {
        double3 cross_test = cross(particles[i].vel, particles[j].vel);
        std::cout << "cross: " << cross_test << std::endl;    // 输出: particles[id].pos
    }


    // 删除粒子（通过 ID）
    bool removeParticle(int id) {
        auto it = std::find_if(particles.begin(), particles.end(),
            [id](const Particle& p){ return p.id == id; });
        
        if(it != particles.end()) {
            particles.erase(it);
            return true;
        }
        return false;
    }

    // 获取所有粒子 (Python 访问接口)
    const std::vector<Particle>& getParticles() const {
        return particles;
    }

    // 将内部的三维 vector 转换为一个 NumPy 数组返回
    py::array_t<double> normal_to_numpy() const {
        if (Cells.empty() || Cells[0].empty() || Cells[0][0].empty()) {
            throw std::runtime_error("数据为空");
        }
        int dim0 = Cells.size();
        int dim1 = Cells[0].size();
        int dim2 = Cells[0][0].size();

        // 创建一个新的 NumPy 数组，形状为 (dim0, dim1, dim2)
        auto result = py::array_t<double>({dim0, dim1, dim2, 3});
        py::buffer_info buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        // 将三维 vector 中的数据逐层复制到连续内存中
        for (int i = 0; i < dim0; i++) {
            if (Cells[i].size() != static_cast<size_t>(dim1)) {
                throw std::runtime_error("第 i 层的行数不一致");
            }
            for (int j = 0; j < dim1; j++) {
                if (Cells[i][j].size() != static_cast<size_t>(dim2)) {
                    throw std::runtime_error("列数不一致");
                }
                for (int k = 0; k < dim2; k++) {
                    // 计算连续内存中的索引位置
                    ptr[i * (dim1 * dim2) + j * dim2 + k] = Cells[i][j][k].normal[0];
                    ptr[i * (dim1 * dim2) + j * dim2 + k + 1] = Cells[i][j][k].normal[1];
                    ptr[i * (dim1 * dim2) + j * dim2 + k + 2] = Cells[i][j][k].normal[2];
                }
            }
        }
        return result;
    }

    // void particle_react_parallel(){
    //     py::gil_scoped_release release;  // 释放 GIL

    //     int steps = 100000;
    //     for(int step=0; step<steps; ++step){
    //         #pragma omp parallel for
    //         for(Particle &part: particles){
    //             part.pos += part.vel;
    //         }
    //     }
    // }

    void particle_react_parallel(){
        py::gil_scoped_release release;  // 释放 Python GIL
        int steps = 100000;
        // 假设 particles 是 std::vector<Particle>
        for (int step = 0; step < steps; ++step) {
            #pragma omp parallel for
            for (size_t i = 0; i < particles.size(); ++i) {
                particles[i].pos += particles[i].vel;
            }
        }
    }
};




PYBIND11_MODULE(react, m) {
    // 绑定基础类型
    PYBIND11_NUMPY_DTYPE(Cell, id, index, film, normal);

    bind_vec3<double>(m, "double3");
    bind_vec3<int>(m, "int3");
    
    // 绑定粒子类型
    bind_particle(m);
    bind_Cell(m);
    
        // 绑定 Simulation 类
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<int, int, int, int>(), py::arg("seed"), py::arg("ni"),py::arg("nj"),py::arg("nk"))
        .def("initialize", &Simulation::initialize, 
             py::arg("num_particles"), 
             py::arg("temperature"),
             py::arg("E"),
             py::arg("id"),
             py::arg("box_size") = 100.0)
        .def("add_particle", &Simulation::addParticle,
             py::arg("pos"), py::arg("vel"), py::arg("E"), py::arg("id"))
        .def("remove_particle", &Simulation::removeParticle,
             py::arg("id"))
        .def("get_particles", &Simulation::getParticles)
        .def("printParticle", &Simulation::printParticle)
        .def("moveParticle", &Simulation::moveParticle)
        .def("crossTest", &Simulation::crossTest)
        .def("particle_react_parallel", &Simulation::particle_react_parallel)
        .def("getCells", &Simulation::getCells)
        .def("inputCell", &Simulation::inputCell, py::arg("cell"))
        .def("testWorld", &Simulation::testWorld)
        .def("runSimulation", &Simulation::runSimulation)
        .def("normal_to_numpy", &Simulation::normal_to_numpy)
        .def("print_react_table_equation", &Simulation::print_react_table_equation)
        .def("set_all_parameters", &Simulation::set_all_parameters,
            py::arg("react_table_equation"),
            py::arg("react_type_table"),
            py::arg("react_prob_chemical"),
            py::arg("react_yield_p0"),
            py::arg("rn_coeffcients"), "react_table_equation, react_type_table, react_prob_chemical")
        .def("inputParticle", &Simulation::inputParticle, 
            py::arg("pos"),
            py::arg("vel"),
            py::arg("E"),
            py::arg("id"), "pos, vel, E, id");

}
