/*Defines flying material data*/

#include <vector>
#include "Field.h"
#include <random>

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


/*object for sampling random numbers*/
class Rnd {
public:
	//constructor: set initial random seed and distribution limits
	Rnd(): mt_gen{std::random_device()()}, rnd_dist{0,1.0} {}
	// 接受种子作为参数的构造函数
	explicit Rnd(unsigned int seed) : mt_gen{ seed }, rnd_dist{ 0, 1.0 } {}

	double operator() () { return rnd_dist(mt_gen); }
	//double operator() () {return rnd_dist(mt_gen);}

protected:
	std::mt19937 mt_gen;	    //random number generator
	std::uniform_real_distribution<double> rnd_dist;  //uniform distribution
};

extern Rnd rnd;		//tell the compiler that an object of type Rnd called rnd is defined somewhere