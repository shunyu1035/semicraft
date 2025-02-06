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
#include "Cell.h"

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

public:
    // 构造函数：初始化随机数引擎
    Simulation(int seed = 42) : rng(seed) {}



    void inputCell(
        py::array_t<Cell, py::array::c_style> cell
    ) {
        // 获取输入数组信息
        auto cell_buf = cell.request();
        auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

        size_t dim_x = cell.shape(0);
        size_t dim_y = cell.shape(1);
        size_t dim_z = cell.shape(2);
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






void compute_squares(int n) {
    // 设置数据规模
    // const size_t n = 1000000;
    std::vector<double> data(n);
    
    // 填充数据
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<double>(i);
    }
    
    {
        // 释放 GIL 以便 C++ 层并行计算不受 Python 的 GIL 限制
        py::gil_scoped_release release;
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] * data[i];
        }
    }
    
    // 打印前10个计算结果，验证运算正确性
    std::cout << "前10个结果：" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}





// 粒子生成函数实现
std::vector<Particle> initial(int N) {
    std::vector<Particle> particles;
    particles.reserve(N);

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.0, 100.0);  // 位置范围0-100
    std::uniform_real_distribution<double> vel_dist(-1.0, 1.0);   // 速度范围-1~1

    for (int i = 0; i < N; ++i) {
        // 生成随机位置和速度
        double3 pos{pos_dist(gen), pos_dist(gen), pos_dist(gen)};
        double3 vel{vel_dist(gen), vel_dist(gen), vel_dist(gen)};
        
        // 创建粒子，初始能量设为0
        particles.emplace_back(pos, vel, 0.0, i);
    }
    return particles;
}





class MyClass {
public:
    // 将数据存储在 std::vector<std::vector<double>> 中
    std::vector<std::vector<double>> data;

    // 从 NumPy 数组设置数据
    void set_data(py::array_t<double> array) {
        // 获取数组的缓冲区信息
        py::buffer_info buf = array.request();

        // 检查数组是否为二维
        if (buf.ndim != 2) {
            throw std::runtime_error("输入数组必须是二维的");
        }

        // 获取数组的形状
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];

        // 获取指向数据的指针
        double* ptr = static_cast<double*>(buf.ptr);

        // 将数据复制到 std::vector<std::vector<double>>
        data.resize(rows);
        for (size_t i = 0; i < rows; ++i) {
            data[i].assign(ptr + i * cols, ptr + (i + 1) * cols);
        }
    }

    // 打印存储的数据
    void print_data() const {
        for (const auto& row : data) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
};


void inputCell(
    py::array_t<Cell, py::array::c_style> cell
) {
    // 获取输入数组信息
    auto cell_buf = cell.request();
    auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

    const int dim_x = cell.shape(0);
    const int dim_y = cell.shape(1);
    const int dim_z = cell.shape(2);
    std::cout << "inputCell:" << dim_x << '_' << dim_y  << '_' << dim_z << std::endl;
}


PYBIND11_MODULE(react, m) {
    // 绑定基础类型
    PYBIND11_NUMPY_DTYPE(Cell, id, index, film, normal);

    bind_vec3<double>(m, "double3");
    bind_vec3<int>(m, "int3");
    
    // 绑定粒子类型
    bind_particle(m);
    bind_Cell(m);
    
    // 暴露初始化函数
    m.def("initial", &initial, py::arg("N"), 
          "Initialize N particles with random positions and velocities");

        // 绑定 Simulation 类
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<int>(), py::arg("seed") = 42)
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
        .def("normal_to_numpy", &Simulation::normal_to_numpy);

    m.def("compute_squares", &compute_squares, "在 C++ 中创建数据并并行计算每个元素的平方，结果直接打印");

    py::class_<MyClass>(m, "MyClass")
    .def(py::init<>())
    .def("set_data", &MyClass::set_data, "从 NumPy 数组设置数据")
    .def("print_data", &MyClass::print_data, "打印存储的数据");

    m.def("inputCell", &inputCell,
        py::arg("cell"));
}
