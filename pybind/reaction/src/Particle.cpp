/*definitions for species functions*/
#include <math.h>
#include <thread>
#include "Particle.h"
#include "Field.h"

// const double PI = 3.14159265358979323846;

// Rnd rnd;

int find_max_position(const std::vector<double>& arr) noexcept {
    // 使用 std::max_element 找到最大值的迭代器
    auto max_iter = std::max_element(arr.begin(), arr.end());

    // 计算迭代器的位置
    int max_pos = std::distance(arr.begin(), max_iter);

    return max_pos;
}

void advanceKernel(size_t p_start, size_t p_end, World &world, std::vector<Particle> &particles, int threadID)
{
	/*loop over particles in [p_start,p_end)*/
	for (size_t p = p_start; p<p_end; p++)
	{
		thread_local Rnd rnd;
		bool react = false;
		Particle &part = particles[p];
		int3 posInt = {(int)part.pos[0], (int)part.pos[1], (int)part.pos[2]};
		// std::cout << "posInt: " << posInt[0] << ", " << posInt[1] << ", " << posInt[2] << std::endl;

		/*判断粒子是否进入cell表面*/
		if (world.inFilm(posInt))
		{
			double angle_rad = acos(fabs(dot(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal)));
			std::vector<int> sticking_acceptList = world.sticking_probability_structed(part, world.Cells[posInt[0]][posInt[1]][posInt[2]], angle_rad, rnd);
			
			// std::cout << "sticking_acceptList: ";
			// for (size_t f = 0; f < sticking_acceptList.size(); ++f) {
			//     std::cout << sticking_acceptList[f] << ' ';
			// }
			// std::cout << '\n';
			bool stick_bool = false;
			for (int s=0; s<world.FILMSIZE; ++s){
				if(sticking_acceptList[s] == 1){
					stick_bool = true;
				}
			}

			// if (part.id==1 && stick_bool == true){
			// 	std::cout << "sticking_acceptList: ";
			// 	for (size_t f = 0; f < sticking_acceptList.size(); ++f) {
			// 		std::cout << sticking_acceptList[f] << ' ';
			// 	}
			// 	std::cout << '\n';
			// }


			if(stick_bool){
				std::vector<double> react_choice_random(world.FILMSIZE, 0);
				for (int i=0; i<world.FILMSIZE; ++i){
					if(sticking_acceptList[i]){
						react_choice_random[i] = rnd();
					}
				}
				int react_choice = find_max_position(react_choice_random);

				// std::cout << "react_choice: " << react_choice << std::endl;

				const int react_type = world.react_type_table[part.id][react_choice];
				
				std::vector<int> react_add(world.FILMSIZE, 0);

				for (int f=0; f<world.FILMSIZE; ++f){
					// std::cout << world.react_table_equation[part.id][react_choice][f] <<  std::endl;
					react_add[f] = world.react_table_equation[part.id][react_choice][f];
					// std::cout << react_add[f] <<  std::endl;
				}
				// chemical transfer
				if(react_type == 1){
					world.film_add(posInt, react_add);
					react = true;
				}
				// chemical remove
				else if(react_type == 4){
					world.film_add(posInt, react_add);
					react = true;
					if (world.film_empty(posInt)) {
						world.update_film_etch_buffers[threadID].push_back(posInt);
					}
				}
				// physics sputter
				else if(react_type == 2){
					double react_yield = world.sputter_yield(world.react_yield_p0[react_choice], angle_rad, part.E, world.film_eth[react_choice]);
					// std::cout << react_yield <<  std::endl;
					if(react_yield > rnd()){
						world.film_add(posInt, react_add);
						react = true;
						if (world.film_empty(posInt)) {
							world.update_film_etch_buffers[threadID].push_back(posInt);
						}
					}
				}
			}

			if (react == false) {
				// if (part.id == world.ArgonID){
				// 	part.E -= world.E_decrease;
				// }
				part.E -= world.E_decrease;
				// std::cout << "reflect before vel: " << p << part.vel << std::endl;
				// std::cout << "reflect normal: " << p << world.Cells[posInt[0]][posInt[1]][posInt[2]].normal << std::endl;
				if (world.reflect_coefficient < rnd()){
					part.vel = world.DiffusionReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal, rnd);
				}
				else{
					part.vel = world.SpecularReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal);
				}
				// part.vel = world.SpecularReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal);
				// part.vel = world.DiffusionReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal, rnd);
				// std::cout << "reflect after vel: " << p << part.vel << std::endl;
			}
		}

		/*update position from v=dx/dt*/
		part.pos += part.vel;

		part.pos = world.mirror(part.pos);
		/*did this particle get inside the sphere or leave the domain?*/
		if (!world.inBounds(part.pos) || react == true || part.E <= 0)
		{
			part = world.inletParticle(rnd);
			continue;
		}
	}
}

void Species::advance(int reaction_count){

	/*calculate number of particles per thread*/
	size_t np = particles.size();
	int n_threads = world.getNumThreads();
	size_t np_per_thread = np/n_threads;
	std::vector<std::thread> threads;
	for (int i=0;i<n_threads;i++) {
		size_t p_start = i*np_per_thread;
		size_t p_end = p_start + np_per_thread;
		if (i==n_threads-1) p_end = np;	//make sure all particles are captured
		threads.emplace_back(advanceKernel, p_start, p_end,	std::ref(world), std::ref(particles), i);
	}

	//wait for threads to finish
	for (std::thread &t: threads) t.join();

	// std::cout << "update_Cells: " << world.update_film_etch.size() << std::endl;

    // std::cout << "update_film_etch: ";
    // for (size_t f = 0; f < world.update_film_etch.size(); ++f) {
    //     std::cout << world.update_film_etch[f] << '\n';
    // }
    // std::cout << '\n';

	for (int threadID=0; threadID<n_threads; threadID++){
		size_t np = world.update_film_etch_buffers[threadID].size();
		for (size_t pt=0; pt<np; pt++){
			world.update_film_etch.push_back(world.update_film_etch_buffers[threadID][pt]);
			world.update_film_etch_buffers[threadID].resize(0);
		}
	}

	world.update_Cells();
	/*perform a particle removal step, dead particles are replaced by the entry at the end*/
	// for (size_t p=0;p<np;p++)
	// {
	// 	// if (particles[p].id<0){
	// 	// 	// inletParticle(particles[p]);
	// 	// 	// std::cout << "Advance reaction_count: "<< reaction_count <<  std::endl;
	// 	// 	reaction_count++;
	// 	// }
	// 	if (particles[p].id>=0){
	// 		reaction_count++;
	// 		continue;
	// 	} 	//ignore live particles
	// 	particles[p] = particles[np-1]; //fill the hole
	// 	np--;	//reduce count of valid elements
	// 	p--;	//decrement p so this position gets checked again
	// }

	//now delete particles[np:end]
	// particles.erase(particles.begin()+np,particles.end());

	// for (int i=0; i<reaction_count;i++){
	// 	addParticleIn();
	// }

}


/*adds a new particle, rewinding velocity by half dt*/
void Species::addParticle(double3 pos, double3 vel, double E, int id)
{
	//don't do anything (return) if pos outside domain bounds [x0,xd)
	if (!world.inBounds(pos)) return;

    //add to list
    particles.emplace_back(pos, vel, E, id);
}


void Species::addParticleIn(){
	Rnd rng; 
	int randID;
	randID = rng.getInt(particleIn.size());

	double3 pos = world.posInlet(rng);
	double3 vel = particleIn[randID].vel;
	double E = particleIn[randID].E;
	int id = particleIn[randID].id;

	// particles.emplace_back(pos, vel, E, id)
	addParticle(pos, vel, E, id);

}
// void Species::change_cell(int idx, int idy, int idz){
//     double3 test{1, 1, 1};
//     world.Cells[idx][idy][idz].normal += test;
// }

// void Species::advance(){
//     std::cout << "advance: " <<  std::endl;
// }