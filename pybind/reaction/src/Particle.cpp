/*definitions for species functions*/
#include <math.h>
#include <thread>
#include "Particle.h"
#include "Field.h"



void advanceKernel(size_t p_start, size_t p_end, World &world, std::vector<Particle> &particles)
{
	/*loop over particles in [p_start,p_end)*/
	for (size_t p = p_start; p<p_end; p++)
	{
		Particle &part = particles[p];

		/*update position from v=dx/dt*/
		part.pos += part.vel;

		/*did this particle get inside the sphere or leave the domain?*/
		if (!world.inBounds(part.pos))
		{
			part.id = -1;	//mark the particle as dead by setting its weight to zero
			continue;
		}
	}
}

void Species::advance(){

	/*calculate number of particles per thread*/
	size_t np = particles.size();
	int n_threads = world.getNumThreads();
	size_t np_per_thread = np/n_threads;
	std::vector<std::thread> threads;
	for (int i=0;i<n_threads;i++) {
		size_t p_start = i*np_per_thread;
		size_t p_end = p_start + np_per_thread;
		if (i==n_threads-1) p_end = np;	//make sure all particles are captured
		threads.emplace_back(advanceKernel, p_start, p_end,	std::ref(world), std::ref(particles));
	}

	//wait for threads to finish
	for (std::thread &t: threads) t.join();

	/*perform a particle removal step, dead particles are replaced by the entry at the end*/
	for (size_t p=0;p<np;p++)
	{
		if (particles[p].id>=0) continue;	//ignore live particles
		particles[p] = particles[np-1]; //fill the hole
		np--;	//reduce count of valid elements
		p--;	//decrement p so this position gets checked again
	}

	//now delete particles[np:end]
	particles.erase(particles.begin()+np,particles.end());

}


/*adds a new particle, rewinding velocity by half dt*/
void Species::addParticle(double3 pos, double3 vel, double E, int id)
{
	//don't do anything (return) if pos outside domain bounds [x0,xd)
	if (!world.inBounds(pos)) return;

    //add to list
    particles.emplace_back(pos, vel, E, id);
}


// void Species::change_cell(int idx, int idy, int idz){
//     double3 test{1, 1, 1};
//     world.Cells[idx][idy][idz].normal += test;
// }

// void Species::advance(){
//     std::cout << "advance: " <<  std::endl;
// }