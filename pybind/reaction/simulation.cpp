#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include <iostream>
#include <memory>
#include <array>
#include <cmath>

namespace py = pybind11;

// 使用 std::array<double,3> 作为 double3 类型
using double3 = std::array<double, 3>;

// World 类：仅用于内部实现，不绑定给 Python
class World {
public:
    World(int ni, int nj, int nk)
        : ni(ni), nj(nj), nk(nk),
          xm({static_cast<double>(ni), static_cast<double>(nj), static_cast<double>(nk)}) {}

    double3 getXm() const { return xm; }

    bool inBounds(double3 pos) {
        for (int i = 0; i < 3; ++i)
            if (pos[i] < 0 || pos[i] >= xm[i])
                return false;
        return true;
    }

    double3 pos(int i, int j, int k) {
        return {static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)};
    }

    const int ni, nj, nk;
protected:
    double3 xm;
};

// 假定 Particle 与 Cell 类的定义
// class Particle {};

/** Data structures for particle storage **/
struct Particle
{
	double3 pos;			/*position*/
	double3 vel;			/*velocity*/
	double E;
    int id;
	Particle(double3 x, double3 v, double E, int id):
	pos{x}, vel{v}, E{E}, id{id} { }
};


// class Cell {};
struct Cell {
    int id;
    std::array<int, 3> index;
    std::array<int, 5> film;
    std::array<double, 3> normal;
};



class Simulation {
private:
    std::vector<Particle> particles;
    std::mt19937 rng;  // 随机数引擎
    std::vector<std::vector<std::vector<Cell>>> Cells;
    // 内部使用 World 对象，不对外暴露
    std::unique_ptr<World> world_ptr;
    int cell_x;
    int cell_y;
    int cell_z;

    // 生成麦克斯韦分布速度（内部使用）
    double3 maxwell_velocity(double T) {
        std::normal_distribution<double> dist(0.0, std::sqrt(T));
        return {dist(rng), dist(rng), dist(rng)};
    }

public:
    // 构造函数
    Simulation(int seed, int cell_x, int cell_y, int cell_z)
        : rng(seed), cell_x(cell_x), cell_y(cell_y), cell_z(cell_z) {}

    // inputCell 方法：从 numpy 数组中构造 cell 数据，并根据尺寸动态创建 World 对象
    void inputCell(py::array_t<Cell, py::array::c_style> cell) {
        auto cell_buf = cell.request();
        auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);

        int dim_x = cell.shape(0);
        int dim_y = cell.shape(1);
        int dim_z = cell.shape(2);

        // 动态创建 World 对象，仅在 Simulation 内部使用
        world_ptr = std::make_unique<World>(dim_x, dim_y, dim_z);
        std::cout << "inputCell: " << dim_x << "_" << dim_y << "_" << dim_z << std::endl;

        Cells.resize(dim_x);
        for (size_t i = 0; i < static_cast<size_t>(dim_x); ++i) {
            Cells[i].resize(dim_y);
            for (size_t j = 0; j < static_cast<size_t>(dim_y); ++j) {
                size_t offset = (i * dim_y + j) * dim_z;
                Cells[i][j].assign(cell_ptr + offset, cell_ptr + offset + dim_z);
            }
        }
    }
};

PYBIND11_MODULE(simulation, m) {
    m.doc() = "pybind11 bindings for Simulation (with hidden World class)";

    // 仅绑定 Simulation 类，不对 World 进行任何绑定
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<int, int, int, int>(),
             py::arg("seed"), py::arg("cell_x"), py::arg("cell_y"), py::arg("cell_z"))
        .def("inputCell", &Simulation::inputCell);
}
