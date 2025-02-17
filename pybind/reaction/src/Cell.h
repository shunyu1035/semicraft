#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cmath> // 包含 M_PI 常量
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include "Field.h"


// /*object for sampling random numbers*/
// class Rnd {
// 	public:
// 		//constructor: set initial random seed and distribution limits
// 		Rnd(): mt_gen{std::random_device()()}, rnd_dist{0,1.0} {}
// 		// 接受种子作为参数的构造函数
// 		explicit Rnd(unsigned int seed) : mt_gen{ seed }, rnd_dist{ 0, 1.0 } {}
	
// 		double operator() () { return rnd_dist(mt_gen); }
// 		//double operator() () {return rnd_dist(mt_gen);}
		
// 	    // 生成 [0, N] 之间的 int 随机数
// 		int getInt(int N) {
// 			std::uniform_int_distribution<int> dist(0, N);  // 指定范围
// 			return dist(mt_gen);
// 		}
// 	protected:
// 		std::mt19937 mt_gen;	    //random number generator
// 		std::uniform_real_distribution<double> rnd_dist;  //uniform distribution
// 	};
	
// extern Rnd rnd;		//tell the compiler that an object of type Rnd called rnd is defined somewhere



// namespace py = pybind11;

// Rnd rnd;

struct Cell {
    int typeID;
    int3 index;
    double3 normal;
    std::vector<int> film;
	Cell(int typeID, int3 index, double3 normal, std::vector<int> film):
	typeID{typeID}, index{index}, normal{normal}, film{film} { }
};


/*defines the computational domain*/
class World
{
public:	
	/*constructor, allocates memory*/
	World(int ni, int nj, int nk, int FILMSIZE): 
	rng(), ni(ni), nj(nj), nk(nk), FILMSIZE(FILMSIZE), xm({(double)ni,(double)nj,(double)nk}), rn_angle(180) {
		for (int i = 0; i < 180; ++i) {
            rn_angle[i] = (M_PI / 2) * i / 179;
        }

		/*set inlet thick here*/
		topGap = 5;
	 }

	// /*functions to set mesh origin and spacing*/
	// void setExtents() {xm[0] = <double>ni; xm[1] = <double>nj; xm[2] = <double>nk;}
	
	void inputParticle(std::vector<Particle> particleAll){
		particleIn = particleAll;
		std::cout << "particleIn size: " << particleIn.size() <<  std::endl;
	}



	Particle inletParticle(){

		int randID;
		randID = rng.getInt(particleIn.size());

		double3 pos = posInlet();

		// std::cout << "particleIn pos: " << pos <<  std::endl;
		Particle part = particleIn[randID];
		part.pos = pos;
		// double3 vel = particleIn[randID].vel;
		// double E = particleIn[randID].E;
		// int id = particleIn[randID].id;

		// part.pos[0] = pos[0];
		// part.pos[1] = pos[1];
		// part.pos[2] = pos[2];
		// part.vel[0] = vel[0];
		// part.vel[1] = vel[1];
		// part.vel[2] = vel[2];
		// part.E = E;
		// part.id = id;


		return part;
		// addParticle(pos, vel, E, id);

	}

	double3 getXm() const {return double3(xm);}

	bool inBounds(double3 pos) {
		for (int i=0;i<3;i++)
			if (pos[i] < 0 || pos[i]>=xm[i]) return false;
		return true;
	}

	bool inFilm(int3 posInt){
	// 	if (posInt[0] < 0 || posInt[0] >= Cells.size() ||
    //     posInt[1] < 0 || posInt[1] >= Cells[0].size() ||
    //     posInt[2] < 0 || posInt[2] >= Cells[0][0].size()) {
    //     return false; // 超出边界，返回 false
    // }
		// int3 posInt = {(int)pos[0], (int)pos[1], (int)pos[2]};
		if (Cells[posInt[0]][posInt[1]][posInt[2]].typeID == 1 ) return true;
		return false;
	}

	// Cell toCell(int3 posInt){
	// 	Cell cell = 
	// 	return Cells[posInt[0]][posInt[1]][posInt[2]];
	// }

    int getNumThreads() const {return num_threads;}

    	/*multithreading support*/
	void setNumThreads(int num_threads) {this->num_threads = num_threads;
		buffers = std::vector<Field>();
		for (int i=0;i<num_threads;i++) buffers.emplace_back(ni,nj,nk);
	}

    void print_cell(){
        int dim_0 = Cells.size();
        int dim_1 = Cells[0].size();
        int dim_2 = Cells[0][0].size();
        std::cout << "World Cell: " << dim_0 << " " << dim_1 << " " << dim_2 <<  std::endl;
    }
    void set_cell(std::vector<std::vector<std::vector<Cell>>> Cell){
        Cells = Cell;
        print_cell();
    }

    void change_cell(int idx, int idy, int idz){
        double3 test{1, 1, 1};
        Cells[idx][idy][idz].normal += test;
    }

    void WprintCell(int idx, int idy, int idz);


	/*another form that takes 3 ints as inputs*/
	double3 pos(int i, int j, int k) {
		double3 x{(double)i,(double)j,(double)k};
		return x;
	}


	void set_parameters(
		const std::vector<std::vector<std::vector<int>>>& react_table_equation,
		const std::vector<std::vector<int>>& react_type_table,
		const std::vector<double>& react_prob_chemical,
		const std::vector<double>& react_yield_p0,
		const std::vector<double>& film_eth,
		const std::vector<std::vector<double>>& rn_coeffcients
	) {
		this->react_table_equation = react_table_equation;
		this->react_type_table = react_type_table;
		this->react_prob_chemical = react_prob_chemical;
		this->react_yield_p0 = react_yield_p0;
		this->film_eth = film_eth;
		this->rn_coeffcients = rn_coeffcients;
		this->rn_matrix = Rn_matrix_func(rn_coeffcients);
		print3DVector(react_table_equation);
	}

	void print3DVector(const std::vector<std::vector<std::vector<int>>>& vec) {
		for (size_t i = 0; i < vec.size(); ++i) {
			std::cout << "react_table_equation " << i << ":\n";
			for (size_t j = 0; j < vec[i].size(); ++j) {
				for (size_t k = 0; k < vec[i][j].size(); ++k) {
					std::cout << vec[i][j][k] << ' ';
				}
				std::cout << '\n';
			}
			std::cout << '\n';
		}
	}

		// 线性插值函数：根据给定的 x 值，在 xp 和 fp 数组中找到合适的区间，然后计算插值
	double linear_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp);

	std::vector<bool> sticking_probability_structed(const Particle& particle, const Cell& cell, double angle_rad, Rnd &rnd);
	
	// 打印 rn_angle 的函数
	void print_rn_angle() const {
		std::cout << "World rn_angle: " <<  std::endl;
		for (const auto& angle : rn_angle) {
			std::cout << angle << ' ';
		}
		std::cout << std::endl;
	}

	double Rn_coeffcient(double c1, double c2, double c3, double c4, double alpha) {
		return c1 + c2 * std::tanh(c3 * alpha - c4);
	}


	// 定义 Rn_probability 函数
	std::vector<double> Rn_probability(const std::vector<double>& c_list) {
		const int i_max = 180;
		// std::vector<double> rn_angle(i_max);
		std::vector<double> rn_prob(i_max);

		// // 使用循环初始化 rn_angle
		// for (int i = 0; i < i_max; ++i) {
		// 	rn_angle[i] = (M_PI / 2) * i / (i_max - 1);
		// }

		// 计算 rn_prob
		for (int i = 0; i < i_max; ++i) {
			rn_prob[i] = Rn_coeffcient(c_list[0], c_list[1], c_list[2], c_list[3], rn_angle[i]);
		}

		// 归一化 rn_prob
		double max_prob = *std::max_element(rn_prob.begin(), rn_prob.end());
		for (int i = 0; i < i_max; ++i) {
			rn_prob[i] /= max_prob;
		}

		// 反转 rn_prob
		for (int i = 0; i < i_max; ++i) {
			rn_prob[i] = 1 - rn_prob[i];
		}

		return rn_prob;
	}

	// 定义 Rn_matrix_func 函数
	std::vector<std::vector<double>> Rn_matrix_func(const std::vector<std::vector<double>>& rn_coeffcients) {
		const int num_rows = rn_coeffcients.size();
		const int num_cols = 180;
		std::vector<std::vector<double>> rn_matrix_f(num_rows, std::vector<double>(num_cols));

		for (int p = 0; p < num_rows; ++p) {
			std::vector<double> rn_prob = Rn_probability(rn_coeffcients[p]);
			for (int pp = 0; pp < num_cols; ++pp) {
				rn_matrix_f[p][pp] = rn_prob[pp];
			}
		}

		return rn_matrix_f;
	}

    // 打印 rn_matrix 的函数
    void print_rn_coeffcients() const {
		std::cout << "print_rn_coeffcients " << ":\n";
        for (const auto& row : rn_coeffcients) {
            for (const auto& val : row) {
                std::cout << val << ' ';
            }
            std::cout << '\n';
        }
		std::cout << std::endl;
    }

    void print_rn_matrix() const {
		std::cout << "print_rn_matrix " << ":\n";
        for (const auto& row : rn_matrix) {
            for (const auto& val : row) {
                std::cout << val << ' ';
            }
            std::cout << '\n';
        }
		std::cout << std::endl;
    }

	// 打印 react_type_table 的函数
	void print_react_type_table() const {
		std::cout << "print_react_type_table " << ":\n";
		for (const auto& row : react_type_table) {
			for (const auto& val : row) {
				std::cout << val << ' ';
			}
			std::cout << '\n';
		}
		std::cout << std::endl;
	}

	double3 posInlet(){
		// Rnd rnd;
		double x = xm[0] * rng();
		double y = xm[1] * rng();
		double z = topGap;
		z *= rng();
		z += xm[2] - topGap;
		double3 pos{x, y, z};
		return pos;
	}

	void film_add(int3 posInt, std::vector<int> react_add);


	// 生成从start到end，步长为step的序列
	std::vector<double> linspace(double start, double end, double step) {
		std::vector<double> result;
		for (double value = start; value <= end; value += step) {
			result.push_back(value);
		}
		return result;
	}

	// 计算溅射产率随入射角度的变化
	void sputter_yield_angle(const double gamma0, const double gammaMax, const double thetaMax) {
		double f = -std::log(gammaMax / gamma0) / (std::log(std::cos(gammaMax)) + 1 - std::cos(thetaMax));
		double s = f * std::cos(thetaMax);
		std::vector<double> theta = linspace(0, M_PI / 2, M_PI / 360);
		std::vector<double> sputterYield(theta.size());

		for (size_t i = 0; i < theta.size(); ++i) {
			sputterYield[i] = gamma0 * std::pow(std::cos(theta[i]), -f) * std::exp(-s * (1 / std::cos(theta[i]) - 1));
		}
		sputterYield.back() = 0;
		theta.back() = M_PI / 2;

		sputterYield_ion.resize(2);
		for (size_t j = 0; j < theta.size(); ++j) {
			sputterYield_ion[0].push_back(sputterYield[j]);
			sputterYield_ion[1].push_back(theta[j]);
		}
	}

	// 计算溅射产率随能量的变化
	double sputter_yield_energy(double E, const double Eth) {
		return std::sqrt(E) - std::sqrt(Eth);
	}

	// 线性插值函数
	double linear_interpolate_sputter(const std::vector<double>& x, const std::vector<double>& y, double xi) {
		if (x.size() != y.size()) {
			throw std::invalid_argument("x and y must have the same size");
		}
		for (size_t i = 1; i < x.size(); ++i) {
			if (xi < x[i]) {
				double t = (xi - x[i - 1]) / (x[i] - x[i - 1]);
				return y[i - 1] + t * (y[i] - y[i - 1]);
			}
		}
		return y.back();
	}

	// 计算最终的溅射产率
	double sputter_yield(double p0, double theta, double energy, const double Eth) {
		double angle_factor = linear_interpolate_sputter(sputterYield_ion[1], sputterYield_ion[0], theta);
		double energy_factor = sputter_yield_energy(energy, Eth);
		return p0 * angle_factor * energy_factor;
	}


	//mesh geometry
	Rnd rng;
	const int ni,nj,nk;	//number of nodes
	const int FILMSIZE;
    std::vector<Field> buffers;	//temporary buffers for density calculation
	std::vector<std::vector<double>> sputterYield_ion;
    std::vector<std::vector<std::vector<Cell>>> Cells;
	
	std::vector<std::vector<int>> react_type_table;
	std::vector<std::vector<std::vector<int>>> react_table_equation;
	std::vector<double> react_yield_p0;
	std::vector<double> film_eth;
	std::vector<Particle> particleIn;	/*contiguous array for add*/
	// Rnd rng;
protected:
	double topGap;
	double3 xm;	//origin-diagonally opposite corner (max bound)
    int num_threads;  //number of threads;
	// std::vector<std::vector<std::vector<int>>> react_table_equation;
	// std::vector<std::vector<int>> react_type_table;
	std::vector<double> react_prob_chemical;
	// std::vector<double> react_yield_p0;
	std::vector<std::vector<double>> rn_coeffcients;
	std::vector<double> rn_angle;
	std::vector<std::vector<double>> rn_matrix;
	std::array<double, 5> react_redepo_sticking = {1.0, 1.0, 1.0, 1.0, 1.0};
	std::array<double, 4> rn_energy = {100, 1000, 1050, 10000};
};













// // 定义网格偏移常量
// constexpr std::array<std::array<int, 3>, 6> GRID_CROSS = {{
//     {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
//     {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
// }};

// // 核心计算函数
// py::array_t<int> update_film_label_index_normal_etch(
//     py::array_t<Cell, py::array::c_style> cell_array,
//     py::array_t<int, py::array::c_style> point_etch,
//     py::array_t<int, py::array::c_style> cellSizeXYZ
// ) {
//     // 获取 NumPy 数组的原始指针
//     auto cell_buf = cell_array.request();
//     auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);
//     auto point_buf = point_etch.request();
//     auto* point_ptr = static_cast<int*>(point_buf.ptr);
    
//     const int num_points = point_etch.shape(0);
//     const int grid_x = cell_array.shape(0);
//     const int grid_y = cell_array.shape(1);
//     const int grid_z = cell_array.shape(2);

//     // 预分配输出数组
//     py::array_t<int> result({num_points, 6, 3});
//     auto res_buf = result.request();
//     int* res_ptr = static_cast<int*>(res_buf.ptr);

//     // OpenMP 并行优化
//     #pragma omp parallel for
//     for (int i = 0; i < num_points; ++i) {
//         const int x = point_ptr[i * 3];
//         const int y = point_ptr[i * 3 + 1];
//         const int z = point_ptr[i * 3 + 2];

//         // 标记当前 Cell
//         cell_ptr[x * grid_y * grid_z + y * grid_z + z].typeID = -1;

//         // 处理 6 个相邻方向
//         for (int j = 0; j < 6; ++j) {
//             const int nx = x + GRID_CROSS[j][0];
//             const int ny = y + GRID_CROSS[j][1];
//             const int nz = z + GRID_CROSS[j][2];

//             // 边界检查
//             if (nx >= 0 && nx < grid_x && 
//                 ny >= 0 && ny < grid_y && 
//                 nz >= 0 && nz < grid_z) {

//                 // 记录邻接点坐标
//                 res_ptr[i * 18 + j * 3] = nx;
//                 res_ptr[i * 18 + j * 3 + 1] = ny;
//                 res_ptr[i * 18 + j * 3 + 2] = nz;

//                 Cell& neighbor = cell_ptr[nx * grid_y * grid_z + ny * grid_z + nz];
                
//                 // 状态更新逻辑
//                 if (neighbor.typeID == 2) {
//                     neighbor.typeID = 1;
//                     // 处理下层邻接点
//                     for (int m = 0; m < 6; ++m) {
//                         const int mx = nx + GRID_CROSS[m][0];
//                         const int my = ny + GRID_CROSS[m][1];
//                         const int mz = nz + GRID_CROSS[m][2];
//                         if (mx >= 0 && mx < grid_x && 
//                             my >= 0 && my < grid_y && 
//                             mz >= 0 && mz < grid_z) {
//                             Cell& sub_neighbor = cell_ptr[mx * grid_y * grid_z + my * grid_z + mz];
//                             if (sub_neighbor.typeID == 3) sub_neighbor.typeID = 2;
//                         }
//                     }
//                 } else if (neighbor.typeID == -1) {
//                     // 处理真空邻接点
//                     for (int l = 0; l < 6; ++l) {
//                         const int lx = nx + GRID_CROSS[l][0];
//                         const int ly = ny + GRID_CROSS[l][1];
//                         const int lz = nz + GRID_CROSS[l][2];
//                         if (lx >= 0 && lx < grid_x && 
//                             ly >= 0 && ly < grid_y && 
//                             lz >= 0 && lz < grid_z) {
//                             Cell& vacuum_neighbor = cell_ptr[lx * grid_y * grid_z + ly * grid_z + lz];
//                             if (vacuum_neighbor.typeID == 1) neighbor.typeID = 0;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     return result;
// }