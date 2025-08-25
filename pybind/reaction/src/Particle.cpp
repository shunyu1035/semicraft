/*definitions for species functions*/
#include <math.h>
#include <thread>
#include "Particle.h"
#include "Field.h"
#include "dda.h"
// const double PI = 3.14159265358979323846;

// Rnd rnd;

int find_max_position(const std::vector<double>& arr) noexcept {
    // 使用 std::max_element 找到最大值的迭代器
    auto max_iter = std::max_element(arr.begin(), arr.end());

    // 计算迭代器的位置
    int max_pos = std::distance(arr.begin(), max_iter);

    return max_pos;
}

int find_max_position_int(const std::vector<int>& arr) noexcept {
    // 使用 std::max_element 找到最大值的迭代器
    auto max_iter = std::max_element(arr.begin(), arr.end());

    // 计算迭代器的位置
    int max_pos = std::distance(arr.begin(), max_iter);

    return max_pos;
}




void advanceKernel(size_t p_start, size_t p_end, int &reaction_count_thread, World &world, std::vector<Particle> &particles, int threadID, bool relax)
{
	/*loop over particles in [p_start,p_end)*/
	for (size_t p = p_start; p<p_end; p++)
	{
		thread_local Rnd rnd;
		bool react = false;
		// bool test = true;
		double reflect_prob = 1.0;
		Particle &part = particles[p];
		int3 posInt = {(int)part.pos[0], (int)part.pos[1], (int)part.pos[2]};
		// std::cout << "posInt: " << posInt[0] << ", " << posInt[1] << ", " << posInt[2] << std::endl;

		/*判断粒子是否进入cell表面*/
		if (world.inFilm(posInt))
		{
			double angle_rad = acos(fabs(dot(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal)));
			std::vector<int> sticking_acceptList = world.sticking_probability_structed(part, world.Cells[posInt[0]][posInt[1]][posInt[2]], angle_rad, rnd);
			
			bool stick_bool = false;
			if (relax) {
				for (int s=0; s<world.FILMSIZE; ++s){
					if(sticking_acceptList[s] == 1){
						stick_bool = true;
					}
				}
			}

			std::vector<double> react_choice_random(world.FILMSIZE, 0);
			for (int i=0; i<world.FILMSIZE; ++i){
				if(sticking_acceptList[i] == 1){
					react_choice_random[i] = rnd();
				}
			}

			// int react_choice = find_max_position(react_choice_random);

			// if (stick == false){
			// 	react_choice = find_max_position_int(react_choice_random);
			// }

			// for (int t=0; t<world.FILMSIZE; ++t){
			// 	if(sticking_acceptList[t] == 1){
			// 		test = false;
			// 	}
			// }

			// if (test){
			// 	std::cout << "react_choice: " << react_choice <<  std::endl;
			// 	std::cout << "cell: " ;
			// 	for (int p=0; p<world.FILMSIZE; ++p){
			// 	std::cout << world.Cells[posInt[0]][posInt[1]][posInt[2]].film[p] <<  " ";
			// 	}
			// 	std::cout <<"\n" << std::endl;

			// 	std::cout << "sticking_acceptList: " ;
			// 	for (int p=0; p<world.FILMSIZE; ++p){
			// 	std::cout << sticking_acceptList[p] <<  " ";
			// 	}
			// 	std::cout <<"\n" << std::endl;

			// 	std::cout << "react_choice_random: " ;
			// 	for (int p=0; p<world.FILMSIZE; ++p){
			// 	std::cout << react_choice_random[p] <<  " ";
			// 	}
			// 	std::cout <<"\n" << std::endl;
			// }

			if(stick_bool){
				
				std::vector<double> react_choice_random(world.FILMSIZE, 0);
				for (int i=0; i<world.FILMSIZE; ++i){
					if(sticking_acceptList[i] == 1){
						react_choice_random[i] = rnd();
					}
				}

				// std::cout << react_choice <<  std::endl;
				int react_choice = find_max_position(react_choice_random);

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
						// world.update_film_etch_buffers[threadID].push_back(posInt);
						world.update_Cells_inthread(posInt);
					}
				}
				// physics sputter
				else if(react_type == 2){
					double react_yield = world.sputter_yield(react_choice, world.react_yield_p0[react_choice], angle_rad, part.E, world.film_eth[react_choice]);
					// std::cout << react_yield <<  std::endl;
					if(react_yield > rnd()){
						world.film_add(posInt, react_add);

						react = true;
						if (world.redepo == true) {
							part.id = react_choice;
							react = false;
						}
						// part.id = react_choice;
						// react = false;
						if (world.film_empty(posInt)) {
							// world.update_film_etch_buffers[threadID].push_back(posInt);
							world.update_Cells_inthread(posInt);
						}
					}
				}

				// depo
				else if(react_type == 3){

					react = true;

					if (world.diffusion == true) {
						for (int df = world.diffusion_distant; df > 0; df--) {
							posInt = world.surface_diffusion(posInt, rnd);
						}
					}

					if (world.film_full(posInt)) {

						int3 depo_cell = world.find_depo_cell(posInt, rnd);

						world.film_add(depo_cell, react_add);

						if (world.film_full(depo_cell)) {
							world.update_Cells_inthread_depo(depo_cell);
						}

					}
					else{
						world.film_add(posInt, react_add);
					}
				}
			}
			// else{
			// 	reflect_prob = world.reflect_probability[part.id][react_choice];
			// }

			if (react == false) {

				int reflect_film = find_max_position_int(world.Cells[posInt[0]][posInt[1]][posInt[2]].film);
				part.E -= world.E_decrease[part.id][reflect_film];

				const double reflect_prob = world.reflect_probability[part.id][reflect_film];
				const double reflect_mode = world.reflect_coefficient[part.id][reflect_film];
				// std::cout <<"partid: " << part.id << ";  reflect_film: " << reflect_film << ";  reflect_prob: " << reflect_prob << std::endl;
				if (reflect_prob > rnd()) {

					if (reflect_mode < rnd()){
						part.vel = world.DiffusionReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal, rnd);
					}
					else{
						part.vel = world.SpecularReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal);
					}
				}
				else{
					react = true;
				}

			}
		}

		/*update position from v=dx/dt*/
		part.pos += part.vel;

		part.pos = world.mirror(part.pos);
		/*did this particle get inside the sphere or leave the domain?*/
		if (!world.inBounds(part.pos) || react == true || part.E <= 0)
		{
			part = world.inletParticle(rnd);
			// continue;
		}

		// count reaction particles
		if (react == true)
		{
			reaction_count_thread++;
			// continue;
		}

	}
}

void advanceKernelDDA(size_t p_start, size_t p_end, int &reaction_count_thread, World &world, std::vector<Particle> &particles, int threadID)
{
	/*loop over particles in [p_start,p_end)*/
	for (size_t p = p_start; p<p_end; p++)
	{
		thread_local Rnd rnd;
		bool react = false;
		// bool test = true;
		double reflect_prob = 1.0;
		Particle &part = particles[p];

		// jumpToTopPeriodic(world.ni, world.nj, world.nk, part.pos, part.vel, world.top);

		// int3 posInt = {(int)part.pos[0], (int)part.pos[1], (int)part.pos[2]};
		// std::cout << "posInt: " << posInt[0] << ", " << posInt[1] << ", " << posInt[2] << std::endl;

		/*判断粒子是否进入cell表面*/
		while (voxelDDA(world.Cells, world.ni, world.nj, world.nk, part.pos, part.vel) || react == false)
		{
			int3 posInt = {(int)part.pos[0], (int)part.pos[1], (int)part.pos[2]};

			if (posInt[0] < 0 || posInt[0] >= world.ni ||
				posInt[1] < 0 || posInt[1] >= world.nj ||
				posInt[2] < 0 || posInt[2] >= world.nk) {
				std::cerr << "posInt out of bounds: "
						<< posInt[0] << " " << posInt[1] << " " << posInt[2] << std::endl;
				continue; // 或 return
			}

			double angle_rad = acos(fabs(dot(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal)));
			std::vector<int> sticking_acceptList = world.sticking_probability_structed(part, world.Cells[posInt[0]][posInt[1]][posInt[2]], angle_rad, rnd);

			bool stick_bool = false;
			for (int s=0; s<world.FILMSIZE; ++s){
				if(sticking_acceptList[s] == 1){
					stick_bool = true;
				}
			}

			std::vector<double> react_choice_random(world.FILMSIZE, 0);
			for (int i=0; i<world.FILMSIZE; ++i){
				if(sticking_acceptList[i] == 1){
					react_choice_random[i] = rnd();
				}
			}


			if(stick_bool){
				
				std::vector<double> react_choice_random(world.FILMSIZE, 0);
				for (int i=0; i<world.FILMSIZE; ++i){
					if(sticking_acceptList[i] == 1){
						react_choice_random[i] = rnd();
					}
				}

				// std::cout << react_choice <<  std::endl;
				int react_choice = find_max_position(react_choice_random);

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
						// world.update_film_etch_buffers[threadID].push_back(posInt);
						world.update_Cells_inthread_DDA(posInt);
					}
				}
				// physics sputter
				else if(react_type == 2){
					double react_yield = world.sputter_yield(react_choice, world.react_yield_p0[react_choice], angle_rad, part.E, world.film_eth[react_choice]);
					// std::cout << react_yield <<  std::endl;
					if(react_yield > rnd()){
						world.film_add(posInt, react_add);

						react = true;
						if (world.redepo == true) {
							part.id = react_choice;
							react = false;
						}
						// part.id = react_choice;
						// react = false;
						if (world.film_empty(posInt)) {
							// world.update_film_etch_buffers[threadID].push_back(posInt);
							world.update_Cells_inthread_DDA(posInt);
						}
					}
				}

				// depo
				else if(react_type == 3){

					react = true;

					if (world.diffusion == true) {
						for (int df = world.diffusion_distant; df > 0; df--) {
							posInt = world.surface_diffusion(posInt, rnd);
						}
					}

					if (world.film_full(posInt)) {

						int3 depo_cell = world.find_depo_cell(posInt, rnd);

						world.film_add(depo_cell, react_add);

						if (world.film_full(depo_cell)) {
							world.update_Cells_inthread_depo_DDA(depo_cell);
						}

					}
					else{
						world.film_add(posInt, react_add);
					}
				}
			}
			// else{
			// 	reflect_prob = world.reflect_probability[part.id][react_choice];
			// }

			if (react == false) {

				int reflect_film = find_max_position_int(world.Cells[posInt[0]][posInt[1]][posInt[2]].film);
				part.E -= world.E_decrease[part.id][reflect_film];

				const double reflect_prob = world.reflect_probability[part.id][reflect_film];
				const double reflect_mode = world.reflect_coefficient[part.id][reflect_film];
				// std::cout <<"partid: " << part.id << ";  reflect_film: " << reflect_film << ";  reflect_prob: " << reflect_prob << std::endl;
				if (reflect_prob > rnd()) {

					if (reflect_mode < rnd()){
						part.vel = world.DiffusionReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal, rnd);
					}
					else{
						part.vel = world.SpecularReflect(part.vel, world.Cells[posInt[0]][posInt[1]][posInt[2]].normal);
					}
				}
				else{
					react = true;
				}

			}
		}


		// count reaction particles
		if (react == true) reaction_count_thread++;

	}
}


int Species::advance_DDA(int &reaction_count, int depo_or_etch, int stopPointY, int stopPointZ){

	/*calculate number of particles per thread*/
	size_t np = particles.size();
	int n_threads = world.getNumThreads();
	// int reaction_count_thread = 0;
	size_t np_per_thread = np/n_threads;

	std::vector<int> reaction_count_thread; 
	reaction_count_thread.resize(n_threads);

	std::vector<std::thread> threads;
	for (int i=0;i<n_threads;i++) {
		size_t p_start = i*np_per_thread;
		size_t p_end = p_start + np_per_thread;
		if (i==n_threads-1) p_end = np;	//make sure all particles are captured
		threads.emplace_back(advanceKernelDDA, p_start, p_end,	std::ref(reaction_count_thread[i]), std::ref(world), std::ref(particles), i);
	}


	//wait for threads to finish
	for (std::thread &t: threads) t.join();

	//count reactions
	for (int i=0;i<n_threads;i++) {
		reaction_count += reaction_count_thread[i];
	}

	/*perform a particle removal step, dead particles are replaced by the entry at the end*/
	for (size_t p=0;p<np;p++)
	{
		particles[p] = world.inletParticle(rnd);
	}

	if(depo_or_etch == -1){
		if(world.scan_stopPoint(stopPointY, stopPointZ)){
			return 1;
		}
	}
	else if(depo_or_etch == 1){
		if(world.scan_stopPoint_depo(stopPointY, stopPointZ)){
			return 1;
		}
	}
}

void Species::advance(int &reaction_count, bool relax){

	// std::cout << "advace ;"<< reaction_count << std::endl;
	/*calculate number of particles per thread*/
	size_t np = particles.size();
	int n_threads = world.getNumThreads();
	// int reaction_count_thread = 0;
	size_t np_per_thread = np/n_threads;

	std::vector<int> reaction_count_thread; 
	reaction_count_thread.resize(n_threads);

	std::vector<std::thread> threads;
	for (int i=0;i<n_threads;i++) {
		size_t p_start = i*np_per_thread;
		size_t p_end = p_start + np_per_thread;
		if (i==n_threads-1) p_end = np;	//make sure all particles are captured
		threads.emplace_back(advanceKernel, p_start, p_end,	std::ref(reaction_count_thread[i]), std::ref(world), std::ref(particles), i, relax);
	}

	// std::cout << "reaction_count_thread [" << 0 << "]: " << reaction_count_thread[0] <<  std::endl;
	//wait for threads to finish
	for (std::thread &t: threads) t.join();

	// std::cout << 'reaction count: ' <<  std::endl;
	for (int i=0;i<n_threads;i++) {
		// std::cout << "reaction_count_thread [" << i << "]: " << reaction_count_thread[i] <<  std::endl;
		reaction_count += reaction_count_thread[i];
	}

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
