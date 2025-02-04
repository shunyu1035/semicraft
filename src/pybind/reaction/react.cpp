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






class ParticleSystem {
private:
    std::vector<Particle> particles;
    std::mt19937 rng;  // 随机数引擎

    // 生成麦克斯韦分布速度
    double3 maxwell_velocity(double T) {
        std::normal_distribution<double> dist(0.0, sqrt(T));
        return {dist(rng), dist(rng), dist(rng)};
    }

public:
    // 构造函数：初始化随机数引擎
    ParticleSystem(int seed = 42) : rng(seed) {}

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
};











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





PYBIND11_MODULE(react, m) {
    // 绑定基础类型
    bind_vec3<double>(m, "double3");
    bind_vec3<int>(m, "int3");
    
    // 绑定粒子类型
    bind_particle(m);
    
    // 暴露初始化函数
    m.def("initial", &initial, py::arg("N"), 
          "Initialize N particles with random positions and velocities");

        // 绑定 ParticleSystem 类
    py::class_<ParticleSystem>(m, "ParticleSystem")
        .def(py::init<int>(), py::arg("seed") = 42)
        .def("initialize", &ParticleSystem::initialize, 
             py::arg("num_particles"), 
             py::arg("temperature"),
             py::arg("E"),
             py::arg("id"),
             py::arg("box_size") = 100.0)
        .def("add_particle", &ParticleSystem::addParticle,
             py::arg("pos"), py::arg("vel"), py::arg("E"), py::arg("id"))
        .def("remove_particle", &ParticleSystem::removeParticle,
             py::arg("id"))
        .def("get_particles", &ParticleSystem::getParticles);
}
