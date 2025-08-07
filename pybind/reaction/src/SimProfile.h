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
#include <thread>
#include <csignal>

#include <stdexcept>
#include <cstdlib>
class SimulationError : public std::runtime_error {
public:
    explicit SimulationError(const std::string &msg)
        : std::runtime_error(msg) {}
};



// 定义全局静态标志，用于指示错误状态
static volatile std::sig_atomic_t error_flag = 0;

// 全局信号处理函数（不要作为非静态成员函数）
void globalSignalHandler(int signum) {
    std::cerr << "Signal " << signum << " received.\n";
    error_flag = 1;  // 标记错误发生
    // 注意：不要在这里抛出异常，也不要调用 std::exit()
    std::exit(1);
}

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

// // Cell结构体绑定
// void bind_Cell(py::module &m) {
//     py::class_<Cell>(m, "Cell")
//         .def(py::init<int, std::array<int, 3>, std::array<int, 5>, std::array<double, 3>>())
//         .def_readwrite("id", &Cell::id)
//         .def_readwrite("index", &Cell::index)
//         .def_readwrite("film", &Cell::film)
//         .def_readwrite("normal", &Cell::normal)
//         .def("__repr__", [](const Cell &p) {
//             return "Cell(id=" + std::to_string(p.id) + ")";
//         });
// }

// Cell结构体绑定
// void bind_Cell(py::module &m) {
//     py::class_<Cell>(m, "Cell")
//         .def(py::init<int, std::array<int, 3>, std::array<int, 5>, std::array<double, 3>>())
//         .def_readwrite("id", &Cell::id)
//         .def_readwrite("index", &Cell::index)
//         .def_readwrite("film", &Cell::film)
//         .def_readwrite("normal", &Cell::normal)
//         .def("__repr__", [](const Cell &p) {
//             return "Cell(id=" + std::to_string(p.id) + ")";
//         });
// }

class Simulation {
private:
    std::vector<Particle> particles;
    // std::vector<std::vector<std::vector<Cell>>> Cells;
    std::vector<std::vector<std::vector<int>>> typeID_in;
    std::vector<std::vector<std::vector<double>>> potential_in;
    std::vector<std::vector<std::vector<int3>>> index_in;
    std::vector<std::vector<std::vector<double3>>> normal_in;
    std::vector<std::vector<std::vector<std::vector<int>>>> film_in;
	//mesh geometry
    const int seed;
	const int ni,nj,nk;	//number of nodes
    const int FILMSIZE;
    const int FilmDensity;
    const size_t max_particles;
    std::vector<std::vector<std::vector<int>>> react_table_equation;
    std::vector<std::vector<int>> react_type_table;
    std::vector<std::vector<double>> reflect_probability;
    std::vector<std::vector<double>> reflect_coefficient;
    std::vector<std::vector<double>> react_prob_chemical;
    std::vector<double> react_yield_p0;
    std::vector<std::vector<double>> rn_coeffcients;
    std::vector<std::vector<double>> E_decrease;
    std::vector<std::vector<double>> sputter_yield_coefficient;
    // std::vector<std::vector<double>> E_decrease;
    std::vector<double> film_eth;


public:
    // 构造函数：初始化随机数引擎
    Simulation(int seed, int ni, int nj, int nk, int FILMSIZE, int FilmDensity, size_t max_particles) : seed(seed), ni{ni}, nj{nj}, nk{nk}, FILMSIZE{FILMSIZE}, FilmDensity{FilmDensity}, max_particles{max_particles}{}


    // 声明一个静态变量用于标记错误状态
    static bool need_recompute;

    // 将 signalHandler 改为静态成员函数
    static void signalHandler(int signum) {
        std::cout << "Interrupt signal (" << signum << ") received." << std::endl;
        need_recompute = true;
    }

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

    // 从 NumPy 数组设置数据，要求输入必须为2维数组
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

    // 从 NumPy 数组设置数据，要求输入必须为2维数组
    void set_reflect_probability(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("reflect_probability 输入数组必须是2维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        ssize_t dim1 = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        reflect_probability.resize(dim0);
        std::cout << "reflect_probability: \n" ;
        for (ssize_t i = 0; i < dim0; ++i) {
            reflect_probability[i].resize(dim1);
            for (ssize_t j = 0; j < dim1; ++j) {
                // 计算内存中对应的起始偏移：
                ssize_t offset = i * dim1 + j;
                reflect_probability[i][j] = ptr[offset];

                // std::cout << "reflect_probability" ;
                std::cout << reflect_probability[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // 从 NumPy 数组设置数据，要求输入必须为2维数组
    void set_reflect_coefficient(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("reflect_coefficient 输入数组必须是2维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        ssize_t dim1 = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        reflect_coefficient.resize(dim0);
        std::cout << "reflect_coefficient: \n" ;
        for (ssize_t i = 0; i < dim0; ++i) {
            reflect_coefficient[i].resize(dim1);
            for (ssize_t j = 0; j < dim1; ++j) {
                // 计算内存中对应的起始偏移：
                ssize_t offset = i * dim1 + j;
                reflect_coefficient[i][j] = ptr[offset];


                std::cout << reflect_coefficient[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }


    // 从 NumPy 数组设置数据，要求输入必须为2维数组
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

    // 从 NumPy 数组设置数据，要求输入必须为2维数组
    void set_E_decrease(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("E_decrease 输入数组必须是2维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        ssize_t dim1 = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);

        // std::cout << "E_decrease dim0: " << dim0 << "dim1: " << dim1 << "\n" << std::end;
        // 按三维结构分配 react_table_equation
        E_decrease.resize(dim0);
        std::cout << "E_decrease dim0" ;
        for (ssize_t i = 0; i < dim0; ++i) {
            E_decrease[i].resize(dim1);
            for (ssize_t j = 0; j < dim1; ++j) {
                // 计算内存中对应的起始偏移：
                ssize_t offset = i * dim1 + j;
                E_decrease[i][j] = ptr[offset];

                // std::cout << "E_decrease dim0" ;
                std::cout << E_decrease[i][j] << " ";

            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

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

    void set_film_eth(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("react_yield_p0 输入数组必须是1维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        double* ptr = static_cast<double*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        film_eth.resize(dim0);
        for (ssize_t i = 0; i < dim0; ++i) {
            // 计算内存中对应的起始偏移：
            ssize_t offset = i;
            film_eth[i] = ptr[offset];
        }
    }

    // 从 NumPy 数组设置数据，要求输入必须为三维数组
    void set_react_prob_chemical(py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("react_prob_chemical 输入数组必须是2维的");
        }
        // 获取每一维的大小
        ssize_t dim0 = buf.shape[0];
        ssize_t dim1 = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);

        // 按三维结构分配 react_table_equation
        react_prob_chemical.resize(dim0);
        for (ssize_t i = 0; i < dim0; ++i) {
            react_prob_chemical[i].resize(dim1);
            for (ssize_t j = 0; j < dim1; ++j) {
                // 计算内存中对应的起始偏移：
                ssize_t offset = i * dim1 + j;
                react_prob_chemical[i][j] = ptr[offset];

            }
        }
    }



    void set_all_parameters(py::array_t<int> react_table_equation, 
                            py::array_t<int> react_type_table,
                            py::array_t<double> reflect_probability,
                            py::array_t<double> reflect_coefficient,
                            py::array_t<double> react_prob_chemical,
                            py::array_t<double> react_yield_p0,
                            py::array_t<double> film_eth,
                            py::array_t<double> rn_coeffcients,
                            py::array_t<double> E_decrease){

        set_react_table_equation(react_table_equation);
        set_react_type_table(react_type_table);
        set_reflect_probability(reflect_probability);
        set_reflect_coefficient(reflect_coefficient);
        set_react_prob_chemical(react_prob_chemical);
        set_react_yield_p0(react_yield_p0);
        set_film_eth(film_eth);
        set_rn_coeffcients(rn_coeffcients);
        set_E_decrease(E_decrease);
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


    int runSimulation(int time, int ArgonID, int depo_or_etch, bool redepo,
        bool diffusion, double diffusion_coeffient, int diffusion_distant, int stopPointY, int stopPointZ, double chemical_angle_v1, double chemical_angle_v2);


    void inputCell(
        py::array_t<int> typeID_py,
        py::array_t<double> potential_py,
        py::array_t<int> index_py,
        py::array_t<double> normal_py,
        py::array_t<int> film_py
    ) {
        // 获取输入数组信息
        py::buffer_info  typeID_py_buf = typeID_py.request();
        py::buffer_info  potential_py_buf = potential_py.request();
        py::buffer_info  index_py_buf = index_py.request();
        py::buffer_info  normal_py_buf = normal_py.request();
        py::buffer_info  film_py_buf = film_py.request();

        if (typeID_py_buf.ndim < 1) {
            throw std::runtime_error("typeID_py 数组至少应该是一维");
        }

        int* typeID_ptr = static_cast<int*>(typeID_py_buf.ptr);
        double* potential_ptr = static_cast<double*>(potential_py_buf.ptr);
        int* index_ptr = static_cast<int*>(index_py_buf.ptr);
        double* normal_ptr = static_cast<double*>(normal_py_buf.ptr);
        int* film_ptr = static_cast<int*>(film_py_buf.ptr);

        size_t dim_x = typeID_py.shape(0);
        size_t dim_y = typeID_py.shape(1);
        size_t dim_z = typeID_py.shape(2);
        size_t typeIDN = typeID_py_buf.shape[0];
        size_t indexN = index_py_buf.shape[0];
        size_t normalN = normal_py_buf.shape[0];
        size_t filmN = film_py_buf.shape[3];
        if ((typeIDN != indexN || typeIDN != normalN)) {
            throw std::runtime_error("Cell 维度不一致");
        }
        std::cout << "inputCell: " << dim_x << " " <<  dim_y << " " << dim_z<< std::endl;

        std::vector<int> film;
        // Cells.resize(dim_x);
        typeID_in.resize(dim_x);
        potential_in.resize(dim_x);
        index_in.resize(dim_x);
        normal_in.resize(dim_x);
        film_in.resize(dim_x);
        for(size_t i=0; i<dim_x; ++i){
            // Cells[i].resize(dim_y);
            typeID_in[i].resize(dim_y);
            potential_in[i].resize(dim_y);
            index_in[i].resize(dim_y);
            normal_in[i].resize(dim_y);
            film_in[i].resize(dim_y);
            for (size_t j = 0; j < dim_y; ++j) {
                for (size_t k = 0; k < dim_z; ++k) {
                    size_t offset3 = ((i * dim_y + j) * dim_z + k) * 3;
                    size_t offset = (i * dim_y + j) * dim_z + k;
                    size_t offset_film = ((i * dim_y + j) * dim_z + k) * filmN;
                    double3 normal = {normal_ptr[offset3], normal_ptr[offset3 + 1], normal_ptr[offset3 + 2]};
                    int3 index = {index_ptr[offset3], index_ptr[offset3 + 1], index_ptr[offset3 + 2]};
                    // std::vector<int> film;
                    film.resize(filmN);
                    for (size_t n = 0; n < filmN; ++n) {
                        film[n] = film_ptr[offset_film + n];
                    }
                    // std::cout << "normal: " << normal << std::endl;
                    // std::cout << "index: " << index << std::endl;
                    // std::cout << "film: ";
                    // for (size_t f = 0; f < film.size(); ++f) {
                    //     std::cout << film[f] << ' ';
                    // }
                    // std::cout << '\n';
                    // Cells[i][j].emplace_back(typeID_ptr[offset], index, normal, film);
                    typeID_in[i][j].emplace_back(typeID_ptr[offset]);
                    potential_in[i][j].emplace_back(potential_ptr[offset]);
                    index_in[i][j].emplace_back(index);
                    normal_in[i][j].emplace_back(normal);
                    film_in[i][j].emplace_back(film);
                }
            }
        }

        // std::cout << "CellSize: " << Cells.size() << " " <<  Cells[0].size() << " " << Cells[0][0].size() << std::endl;
        // for(size_t i=0; i<num; ++i){
        //     particles[i] = particle_ptr[i];
        // }
        std::cout << "inputCell_over " << std::endl;
    }


    void input_sputter_yield_coefficient(py::array_t<double> sputter_yield_coefficient_py){
        py::buffer_info  sputter_yield_coefficient_py_buf = sputter_yield_coefficient_py.request();
        int shape = sputter_yield_coefficient_py_buf.shape[0];

        if(shape != FILMSIZE) {
            std::cout << "shape != FILMSIZE "<< std::endl;
        }

        double* sputter_yield_coefficient_ptr = static_cast<double*>(sputter_yield_coefficient_py_buf.ptr);
        sputter_yield_coefficient.resize(FILMSIZE);

        std::cout << "sputter_yield_coefficient: ";
        for(int i=0; i<FILMSIZE; ++i){
            sputter_yield_coefficient[i].resize(3);
            for (int j = 0; j < 3; ++j) {
                int offset = i * 3 + j;
                sputter_yield_coefficient[i][j] = sputter_yield_coefficient_ptr[offset];
                std::cout << sputter_yield_coefficient[i][j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n' << std::endl;

    }

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
    // const std::vector<std::vector<std::vector<Cell>>>& getCells() const {
    //     return Cells;
    // }


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

    // void printCell(int idx, int idy, int idz) {
    //     std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].typeID: " << Cells[idx][idy][idz].typeID << std::endl;
    //     std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].index: " << Cells[idx][idy][idz].index << std::endl;    // 输出: particles[id].pos
    //     std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].normal: " << Cells[idx][idy][idz].normal << std::endl;    // 输出: particles[id].pos
    //     std::cout << "Cell["<< idx <<"]["<< idy <<"]["<< idz <<"].film: " << std::endl;
    //     for (size_t i = 0; i < Cells[idx][idy][idz].film.size(); ++i) {
    //         std::cout << Cells[idx][idy][idz].film[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    

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
        if (normal_in.empty() || normal_in[0].empty() || normal_in[0][0].empty()) {
            throw std::runtime_error("数据为空");
        }
        int dim0 = ni;
        int dim1 = nj;
        int dim2 = nk;

        // 创建一个新的 NumPy 数组，形状为 (dim0, dim1, dim2)
        auto result = py::array_t<double>({dim0, dim1, dim2, 3});
        py::buffer_info buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        // 将三维 vector 中的数据逐层复制到连续内存中
        for (int i = 0; i < dim0; i++) {
            if (normal_in[i].size() != static_cast<size_t>(dim1)) {
                throw std::runtime_error("第 i 层的行数不一致");
            }
            for (int j = 0; j < dim1; j++) {
                if (normal_in[i][j].size() != static_cast<size_t>(dim2)) {
                    throw std::runtime_error("列数不一致");
                }
                for (int k = 0; k < dim2; k++) {
                    // 计算连续内存中的索引位置
                    for (size_t l = 0; l < 3; ++l) {
                        ptr[i * (dim1 * dim2)*3  + j * dim2*3 + k*3 + l] = normal_in[i][j][k](l);
                    }
                    // ptr[i * (dim1 * dim2) + j * dim2 + k + 1] = Cells[i][j][k].normal(1);
                    // ptr[i * (dim1 * dim2) + j * dim2 + k + 2] = Cells[i][j][k].normal(2);
                }
            }
        }
        return result;
    }

    py::tuple cell_data_to_numpy() const {
        if (typeID_in.empty() || typeID_in[0].empty() || typeID_in[0][0].empty()) {
            throw std::runtime_error("Cells 数据为空");
        }
    
        size_t dim0 = typeID_in.size();
        size_t dim1 = typeID_in[0].size();
        size_t dim2 = typeID_in[0][0].size();
    
        // 创建用于存储 typeID 的 NumPy 数组
        auto typeID_array = py::array_t<int>({dim0, dim1, dim2});
        auto typeID_buf = typeID_array.request();
        int* typeID_ptr = static_cast<int*>(typeID_buf.ptr);

        // 创建用于存储 potential 的 NumPy 数组
        auto potential_array = py::array_t<double>({dim0, dim1, dim2});
        auto potential_buf = potential_array.request();
        double* potential_ptr = static_cast<double*>(potential_buf.ptr);

        // 创建用于存储 film 的 NumPy 数组，形状为 (dim0, dim1, dim2, FILMSIZE)
        auto film_array = py::array_t<int>({dim0, dim1, dim2, static_cast<size_t>(FILMSIZE)});
        auto film_buf = film_array.request();
        int* film_ptr = static_cast<int*>(film_buf.ptr);
    
        // 将数据复制到 NumPy 数组
        for (size_t i = 0; i < dim0; ++i) {
            for (size_t j = 0; j < dim1; ++j) {
                for (size_t k = 0; k < dim2; ++k) {
                    // const Cell& cell = Cells[i][j][k];
                    typeID_ptr[i * dim1 * dim2 + j * dim2 + k] = typeID_in[i][j][k];
                    potential_ptr[i * dim1 * dim2 + j * dim2 + k] = potential_in[i][j][k];

                    // std::cout << "potential: " << i << " " << j << " " << k << " " << potential_in[i][j][k] << std::endl;
    
                    // 复制 film 数据，如果长度不足 FILMSIZE，填充 0
                    for (size_t l = 0; l < static_cast<size_t>(FILMSIZE); ++l) {
                        if (l < film_in[i][j][k].size()) {
                            film_ptr[i * dim1 * dim2 * FILMSIZE + j * dim2 * FILMSIZE + k * FILMSIZE + l] = film_in[i][j][k][l];
                        } else {
                            film_ptr[i * dim1 * dim2 * FILMSIZE + j * dim2 * FILMSIZE + k * FILMSIZE + l] = 0;
                        }
                    }
                }
            }
        }
    
        return py::make_tuple(typeID_array, film_array, potential_array);
    } 


    // void print_Cells(){
    //     // 将数据复制到 NumPy 数组

    // int surface = 0;
    // for (size_t i = 0; i < ni; ++i) {
    //     for (size_t j = 0; j < nj; ++j) {
    //         for (size_t k = 0; k < nk; ++k) {
    //             if(Cells[i][j][k].typeID == 1) {
    //                 surface++;
    //                 std::cout << "surface: " << i << " " << j << " " << k << " " << Cells[i][j][k].normal << std::endl;
    //             }
    //         }
    //     }
    // }
    // std::cout << "surfacecount: "<< surface << std::endl;
    // }


    int recompute() {
        return 0;
    }
};


