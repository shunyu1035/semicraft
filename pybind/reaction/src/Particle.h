/*Defines flying material data*/
#include <iostream>
#include <vector>
#include "Field.h"
#include "Cell.h"
#include <random>
#include <algorithm>

// class RndInt {
// 	public:
// 		// 默认构造函数：使用随机设备种子
// 		RndInt() : mt_gen{std::random_device()()} {}
	
// 		// 允许用户指定种子
// 		explicit RndInt(unsigned int seed) : mt_gen{seed} {}
	
// 		// 生成 0 到 N 之间的整数
// 		int operator()(int N) { 
// 			std::uniform_int_distribution<int> dist(0, N);
// 			return dist(mt_gen);
// 		}
	
// 	protected:
// 		std::mt19937 mt_gen;  // Mersenne Twister 伪随机数生成器
// 	};
	
// 	// 定义一个全局对象
// extern RndInt rndInt;


/*species container*/
class Species 
{
public:
	Species(std::string name, int id, World &world) :
	rnd(), name(name), id(id), world(world) { 	}

	/*returns the number of simulation particles*/
	size_t getNp()	{return particles.size();}


	/*adds a new particle*/
	void addParticle(double3 pos, double3 vel, double E, int id);

	void addParticleIn();

	// void inletParticle(Particle part){
	// 	// Rnd rng; 
	// 	int randID;
	// 	randID = rnd.getInt(particles.size());

	// 	double3 pos = world.posInlet();
	// 	double3 vel = particleIn[randID].vel;
	// 	double E = particleIn[randID].E;
	// 	int id = particleIn[randID].id;

	// 	part.pos = pos;
	// 	part.vel = vel;
	// 	part.E = E;
	// 	part.id = id;
	// 	// addParticle(pos, vel, E, id);

	// }

	void inputParticle(std::vector<Particle> particleAll){

		size_t max_particles = 100000;  // 最多取10万
		size_t half_size = particleAll.size() / 2;  // 或者取前一半

		// 实际取 min(half_size, max_particles) 个
		size_t n = std::min(half_size, max_particles);
		
		// 取前 n 个元素
		particles = std::vector<Particle>(particleAll.begin(), particleAll.begin() + n);

		// 全部存到 particleIn（你也可以同样截断）
		particleIn = particleAll;

		std::cout << "particle size: " << particles.size() <<  std::endl;
	}

    void change_cell(int idx, int idy, int idz){
		double3 test{1, 1, 1};
		world.Cells[idx][idy][idz].normal += test;
	}
	
   // print
	void printParticle(int id) {
		int a = particles.size();
		if(a > id){
			std::cout << "particles["<< id <<"].pos: " << particles[id].pos << std::endl;    // 输出: particles[id].pos
			std::cout << "particles["<< id <<"].vel: " << particles[id].vel << std::endl;    // 输出: particles[id].pos
		}
		else{
			std::cout << "No enough particle: " << particleIn.size() <<  std::endl;
		}
	}
	/*moves all particles */
	void advance(int reaction_count);

	void showParticleIn(int id){
		int a = particleIn.size();
		if(a > id){
			std::cout << "particleIn["<< id <<"].pos: " << particleIn[id].pos << std::endl;    // 输出: particles[id].pos
			std::cout << "particleIn["<< id <<"].vel: " << particleIn[id].vel << std::endl;    // 输出: particles[id].pos
			std::cout << "particleIn["<< id <<"].id: " << particleIn[id].id << std::endl; 
			std::cout << "particleIn["<< id <<"].E: " << particleIn[id].E << std::endl;
		}
		else{
			std::cout << "No enough particle: " << particleIn.size() <<  std::endl;
		}
	}
	
	Rnd rnd;  // 类的成员变量
	const std::string name;			/*species name*/
	const int id;
	std::vector<Particle> particles;	/*contiguous array for storing particles*/
	std::vector<Particle> particleIn;	/*contiguous array for add*/

protected:
	World &world;
	// std::vector<Particle> particleIn;	/*contiguous array for add*/
};


