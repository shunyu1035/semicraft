/*definitions for species functions*/
#include <math.h>
#include <thread>
#include "Particle.h"
#include "Field.h"


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

void Species::advance(){
    std::cout << "advance: " <<  std::endl;
}