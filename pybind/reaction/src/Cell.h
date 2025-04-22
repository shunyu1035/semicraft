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
#include <mutex>
#include <atomic>
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

// struct Cell {
//     int typeID;
//     int3 index;
//     double3 normal;
//     std::vector<int> film;
// 	std::mutex film_mutex; // 添加互斥锁
// 	Cell(int typeID, int3 index, double3 normal, std::vector<int> film):
// 	typeID{typeID}, index{index}, normal{normal}, film{film} { }
// };

class Cell {
	public:
		int typeID;
		int3 index;
		double3 normal;
		std::vector<int> film;
		std::mutex film_mutex; // 添加互斥锁
		// 自定义拷贝构造函数
		Cell(const Cell& other) : typeID(other.typeID), index(other.index), normal(other.normal), film(other.film) {}
	
		// 其他构造函数
		Cell(int typeID, int3 index, double3 normal, std::vector<int> film) :
			typeID{typeID}, index{index}, normal{normal}, film{film} { }
	
		// 其他成员函数
	};

/*defines the computational domain*/
class World
{
public:	
	/*constructor, allocates memory*/
	World(int ni, int nj, int nk, int FILMSIZE, int FilmDensity, int ArgonID, double reflect_coefficient,  double chemical_angle_v1, double chemical_angle_v2): 
	rng(), ni(ni), nj(nj), nk(nk), FILMSIZE(FILMSIZE), FilmDensity(FilmDensity), ArgonID(ArgonID), reflect_coefficient(reflect_coefficient),
		chemical_angle_v1(chemical_angle_v1), chemical_angle_v2(chemical_angle_v2), 
		xm({(double)ni,(double)nj,(double)nk}), ijk({ni,nj,nk}), rn_angle(180){
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


    // // 辅助函数：计算镜像索引
    int3 mirror_index(int3 posInt) {

		for(size_t i=0; i<3;++i){
			if (posInt[i] < 0) posInt[i] += ijk[i];
        // 如果索引超出上界，则镜像计算：例如，当 idx == dim 时返回 dim-1
			if (posInt[i] >= ijk[i]) posInt[i] -= ijk[i];
		}
        return posInt;
    }


	Particle inletParticle(Rnd &rnd){

		int randID;
		randID = rnd.getInt(particleIn.size()-1);

		double3 pos = posInlet(rnd);

		// std::cout << "particleIn pos: " << pos <<  std::endl;
		Particle part = particleIn[randID];

		// for debug
		// if (part.id != 0) {
		// 	std::cout << "inlet randID:  "<< randID <<  std::endl;
		// 	std::cout << "inlet part.id:  "<< part.id <<  std::endl;
		// 	std::cout << "inlet part.vel0:  "<< part.vel[0] <<  std::endl;
		// 	std::cout << "inlet part.vel1:  "<< part.vel[1] <<  std::endl;
		// 	std::cout << "inlet part.vel2:  "<< part.vel[2] <<  std::endl;
		// 	std::cout << "inlet part.E:  "<< part.E <<  std::endl;
		// }

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

	double3 mirror(double3 pos) {

		double3 newPos = pos;
		for (int i=0;i<2;i++) {
			if (newPos[i] < 0){
				// std::cout << "mirror : "<< part.pos << std::endl;
				newPos[i] += xm[i];
			}
			else if (newPos[i] > xm[i]){
				// std::cout << "mirror : "<< part.pos << std::endl;
				newPos[i] -= xm[i];
			}
		}
		return newPos;
	}

	bool inFilm(int3 posInt){
	// 	if (posInt[0] < 0 || posInt[0] >= Cells.size() ||
    //     posInt[1] < 0 || posInt[1] >= Cells[0].size() ||
    //     posInt[2] < 0 || posInt[2] >= Cells[0][0].size()) {
    //     return false; // 超出边界，返回 false
    // }
		// int3 posInt = {(int)pos[0], (int)pos[1], (int)pos[2]};
		if (Cells[posInt[0]][posInt[1]][posInt[2]].typeID == 1 || Cells[posInt[0]][posInt[1]][posInt[2]].typeID == 2) return true;
		return false;
	}

	// Cell toCell(int3 posInt){
	// 	Cell cell = 
	// 	return Cells[posInt[0]][posInt[1]][posInt[2]];
	// }

    int getNumThreads() const {return num_threads;}

    	/*multithreading support*/
	void setNumThreads(int num_threads) {
		this->num_threads = num_threads;
		update_film_etch_buffers.resize(num_threads);
		// buffers = std::vector<Field>();
		// for (int i=0;i<num_threads;i++) buffers.emplace_back(ni,nj,nk);
	}

    void print_cell(){
        int dim_0 = Cells.size();
        int dim_1 = Cells[0].size();
        int dim_2 = Cells[0][0].size();
        std::cout << "World Cell: " << dim_0 << " " << dim_1 << " " << dim_2 <<  std::endl;
    }
    // void set_cell(std::vector<std::vector<std::vector<Cell>>> Cell){
    //     Cells = Cell;
    //     print_cell();
    // }

	void set_cell(std::vector<std::vector<std::vector<int>>> typeID_in,
		std::vector<std::vector<std::vector<int3>>> index_in,
		std::vector<std::vector<std::vector<double3>>> normal_in,
		std::vector<std::vector<std::vector<std::vector<int>>>> film_in)
	{
        size_t dim_x = typeID_in.size();
        size_t dim_y = typeID_in[0].size();
        size_t dim_z = typeID_in[0][0].size();

		Cells.resize(dim_x);
		for(size_t i=0; i<dim_x; ++i){
			Cells[i].resize(dim_y);
			for (size_t j = 0; j < dim_y; ++j) {
				for (size_t k = 0; k < dim_z; ++k) {
					// Cells[i][j].emplace_back(typeID_in[i][j][k], index_in[i][j][k], normal_in[i][j][k], film_in[i][j][k]);
					Cell new_cell(typeID_in[i][j][k], index_in[i][j][k], normal_in[i][j][k], film_in[i][j][k]);
					Cells[i][j].push_back(new_cell);
				}
			}
		}
        print_cell();
    }

	std::vector<std::vector<std::vector<int>>> output_typeID_in(){
		size_t dim_x = Cells.size();
        size_t dim_y = Cells[0].size();
        size_t dim_z = Cells[0][0].size();

		std::vector<std::vector<std::vector<int>>> output_typeID;
		output_typeID.resize(dim_x);
		for(size_t i=0; i<dim_x; ++i){
			output_typeID[i].resize(dim_y);
			for (size_t j = 0; j < dim_y; ++j) {
				for (size_t k = 0; k < dim_z; ++k) {
					output_typeID[i][j].emplace_back(Cells[i][j][k].typeID);
				}
			}
		}
		return output_typeID;
	}

	std::vector<std::vector<std::vector<std::vector<int>>>> output_film_in(){
		size_t dim_x = Cells.size();
        size_t dim_y = Cells[0].size();
        size_t dim_z = Cells[0][0].size();

		std::vector<std::vector<std::vector<std::vector<int>>>> output_film;
		output_film.resize(dim_x);
		for(size_t i=0; i<dim_x; ++i){
			output_film[i].resize(dim_y);
			for (size_t j = 0; j < dim_y; ++j) {
				output_film[i][j].resize(dim_z);
				for (size_t k = 0; k < dim_z; ++k) {
					for (int l = 0; l < FILMSIZE; ++l) {
						output_film[i][j][k].emplace_back(Cells[i][j][k].film[l]);
					}
				}
			}
		}
		return output_film;
	}

	std::vector<std::vector<std::vector<double3>>> output_normal_in(){
		size_t dim_x = Cells.size();
        size_t dim_y = Cells[0].size();
        size_t dim_z = Cells[0][0].size();

		std::vector<std::vector<std::vector<double3>>> output_normal;
		output_normal.resize(dim_x);
		for(size_t i=0; i<dim_x; ++i){
			output_normal[i].resize(dim_y);
			for (size_t j = 0; j < dim_y; ++j) {
				// output_normal[i][j].resize(dim_z);
				for (size_t k = 0; k < dim_z; ++k) {
						output_normal[i][j].emplace_back(Cells[i][j][k].normal);
				}
			}
		}
		return output_normal;
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
		const std::vector<std::vector<double>>& react_prob_chemical,
		const std::vector<double>& react_yield_p0,
		const std::vector<double>& film_eth,
		const std::vector<std::vector<double>>& rn_coeffcients,
		const std::vector<std::vector<double>>& E_decrease
	) {
		this->react_table_equation = react_table_equation;
		this->react_type_table = react_type_table;
		this->react_prob_chemical = react_prob_chemical;
		this->react_yield_p0 = react_yield_p0;
		this->film_eth = film_eth;
		this->rn_coeffcients = rn_coeffcients;
		this->E_decrease = E_decrease;
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

	std::vector<int> sticking_probability_structed(const Particle particle, const Cell cell, double angle_rad, Rnd &rnd);
	
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


	// 打印 react_prob_chemical 的函数
	void print_react_prob_chemical() const {
		std::cout << "print_react_prob_chemical " << ":\n";
		for (const auto& row : react_prob_chemical) {
			for (const auto& val : row) {
				std::cout << val << ' ';
			}
			std::cout << '\n';
		}
		std::cout << std::endl;
	}


	void print_react_yield_p0() const {
		std::cout << "print_react_yield_p0 " << ":\n";
		for (const auto& row : react_yield_p0) {
			std::cout << row << ' ';
		}
		std::cout << '\n';
		std::cout << std::endl;
	}

	double3 posInlet(Rnd &rnd){
		// Rnd rnd;
		double x = xm[0] * rnd();
		double y = xm[1] * rnd();
		double z = topGap;
		z *= rnd();
		z += xm[2] - topGap;
		double3 pos{x, y, z};
		return pos;
	}

	// void film_add(int3 posInt, std::vector<int> react_add);
	void film_add(int3 posInt, const std::vector<int>& react_add);

	// 生成从start到end，步长为step的序列
	std::vector<double> linspace(double start, double end, double step) {
		std::vector<double> result;
		for (double value = start; value <= end; value += step) {
			result.push_back(value);
		}
		return result;
	}

	// // 计算溅射产率随入射角度的变化
	// void sputter_yield_angle(const double gamma0, const double gammaMax, const double thetaMax) {
	// 	double f = -std::log(gammaMax / gamma0) / (std::log(std::cos(gammaMax)) + 1 - std::cos(thetaMax));
	// 	double s = f * std::cos(thetaMax);
	// 	std::vector<double> theta = linspace(0, M_PI / 2, M_PI / 360);
	// 	std::vector<double> sputterYield(theta.size(), 0.0);

	// 	for (size_t i = 0; i < theta.size(); ++i) {
	// 		sputterYield[i] = gamma0 * std::pow(std::cos(theta[i]), -f) * std::exp(-s * (1 / std::cos(theta[i]) - 1));
	// 		if (sputterYield[i] < 0.01) {
	// 			break;  // 提前终止计算
	// 		}
	// 	}
	// 	sputterYield.back() = 0;
	// 	theta.back() = M_PI / 2;

	// 	sputterYield_ion.resize(2);
	// 	for (size_t j = 0; j < theta.size(); ++j) {
	// 		sputterYield_ion[0].push_back(sputterYield[j]);
	// 		sputterYield_ion[1].push_back(theta[j]);
	// 	}
	// 	std::cout << "sputter_yield_angle " << ":\n";
	// 	for (const auto& row : sputterYield) {
	// 		std::cout << row << ' ';
	// 	}
	// 	std::cout << '\n';
	// 	std::cout << std::endl;
	// }

	// 计算溅射产率随入射角度的变化
	void sputter_yield_angle(const std::vector<std::vector<double>>& sputter_yield_coefficient) {
		sputterYield_theta = linspace(0, M_PI / 2, M_PI / 360);
		sputterYield_theta.back() = M_PI / 2;
		std::cout << "set sputter_yield_angle " << std::endl;

		sputterYield_ion.resize(FILMSIZE);

		for(int i=0; i<FILMSIZE; ++i){
			double gamma0 = sputter_yield_coefficient[i][0];
			double gammaMax = sputter_yield_coefficient[i][1];
			double thetaMax = sputter_yield_coefficient[i][2];
			double f = -std::log(gammaMax / gamma0) / (std::log(std::cos(gammaMax)) + 1 - std::cos(thetaMax));
			double s = f * std::cos(thetaMax);
		
			std::cout << "gamma0 gammaMax thetaMax: " << gamma0 << " " << gammaMax <<  " "  << thetaMax <<  " :"  << ":\n" << std::endl;

			std::vector<double> sputterYield(sputterYield_theta.size(), 0.0);
			for (size_t t = 0; t < sputterYield_theta.size(); ++t) {
				sputterYield[t] = gamma0 * std::pow(std::cos(sputterYield_theta[t]), -f) * std::exp(-s * (1 / std::cos(sputterYield_theta[t]) - 1));
				if (sputterYield[t] < 0.01) {
					break;  // 提前终止计算
				}
			}
			sputterYield.back() = 0;
			for (size_t j = 0; j < sputterYield_theta.size(); ++j) {
				sputterYield_ion[i].push_back(sputterYield[j]);
			}
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
	double sputter_yield(int react_choice, double p0, double theta, double energy, const double Eth) {
		double angle_factor = linear_interpolate_sputter(sputterYield_theta, sputterYield_ion[react_choice], theta);
		double energy_factor = sputter_yield_energy(energy, Eth);
		return p0 * angle_factor * energy_factor;
	}


	double3 SpecularReflect(double3 vel, double3 normal){
		double3 newVel;
		newVel = vel - 2*dot(vel, normal)*normal;
		return newVel;
	}


	double3 DiffusionReflect(double3 vel, double3 normal, Rnd &rnd){
		double3 newVel;
		double3 Ut;
		Ut = vel - dot(vel, normal)*normal;
		double3 tw1 = unit(Ut);
		double3 tw2 = cross(tw1, normal);
		double pm = 1;
		if( dot(vel, normal) > 0){
			pm = -1;
		}
		newVel = unit(rnd.randn()*tw1 + rnd.randn()*tw2  + pm*sqrt(-2*log((1-rnd())))*normal);
		return newVel;
	}

	bool film_empty(int3 posInt){
		int sum = 0;
		for(int i=0; i<FILMSIZE; ++i){
			sum += Cells[posInt[0]][posInt[1]][posInt[2]].film[i];
		}

		if(sum <= 0){
			return true;
		}
		return false;
	}

	bool film_full(int3 posInt){
		int sum = 0;
		for(int i=0; i<FILMSIZE; ++i){
			sum += Cells[posInt[0]][posInt[1]][posInt[2]].film[i];
		}

		if(sum >= FilmDensity){
			return true;
		}
		return false;
	}

	double react_prob_chemical_angle(double angle_rad);

	void print_Cells();

	void update_Cells();

	void update_Cells_inthread(int3 posInt);

	void get_normal_from_grid(int3 posInt);

	void update_normal_in_matrix();

	void update_normal_in_matrix_inthread(int3 posInt);

	// bool scan_bottom(){
	// 	int filmThickness = 0;
	// 	int centerX = ni/2;
	// 	int centerY = nj/2;
	// 	for(int z=0; z<nk; ++z){
	// 		int sum = 0;
	// 		for(int f=0; f<FILMSIZE; ++f){
	// 			sum += Cells[centerX][centerY][z].film[f];
	// 		}
	// 		if(sum <= 0){
	// 			filmThickness = z;
	// 			break;
	// 		}
	// 	}
	// 	if(filmThickness == bottom){
	// 		return true;
	// 	}
	// 	return false;
	// }

	bool scan_stopPoint(int stopPointY, int stopPointZ){
		int centerX = ni/2;
		int sum = 0;
		for(int f=0; f<FILMSIZE; ++f){
			sum += Cells[centerX][stopPointY][stopPointZ].film[f];
		}

		if(sum <= 0){
			return true;
		}else {
			return false;
		}
		// return false;
	}

	//mesh geometry
	Rnd rng;
	const int ni,nj,nk;	//number of nodes
	const int FILMSIZE;
	const int FilmDensity;
	int ArgonID;
	double reflect_coefficient;
	// double E_decrease;
	// std::vector<double> E_decrease;
	// double chemical_angle_v1;
	// double chemical_angle_v2;
	double chemical_angle_v1;
	double chemical_angle_v2;
	std::vector<std::vector<double>> sputterYield_ion;
	std::vector<double> sputterYield_theta;
	// std::vector<std::vector<std::vector<double>>> sputterYield_ion;
    std::vector<std::vector<std::vector<Cell>>> Cells;
	
	std::vector<std::vector<int>> react_type_table;
	std::vector<std::vector<std::vector<int>>> react_table_equation;
	std::vector<double> react_yield_p0;
	std::vector<double> film_eth;
	std::vector<std::vector<double>> E_decrease;
	std::vector<Particle> particleIn;	/*contiguous array for add*/
	std::vector<int3> update_film_etch;
	std::vector<std::vector<int3>> update_film_etch_buffers;	//temporary buffers for density calculation

	std::vector<int3> grid_cross = {
		{1, 0, 0},
		{-1,0, 0},
		{0, 1, 0},
		{0,-1, 0},
		{0, 0, 1},
		{0, 0,-1}
	};
	// Rnd rng;
protected:
	double topGap;
	double3 xm;	//origin-diagonally opposite corner (max bound)
	int3 ijk; // for mirror correctify
	// int ArgonID;
	// double reflect_coefficient;
    int num_threads;  //number of threads;
	// std::vector<std::vector<std::vector<int>>> react_table_equation;
	// std::vector<std::vector<int>> react_type_table;
	std::vector<std::vector<double>> react_prob_chemical;
	// std::vector<double> react_yield_p0;
	std::vector<std::vector<double>> rn_coeffcients;
	std::vector<double> rn_angle;
	std::vector<std::vector<double>> rn_matrix;
	std::array<double, 11> react_redepo_sticking = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	std::array<double, 4> rn_energy = {100, 1000, 1050, 10000};
};

