#include "SimProfile.h"




void Simulation::runSimulation(int id){
    World world(ni, nj, nk, 5);
    world.print_rn_angle();

    double3 xm = world.getXm();
    std::cout << "World xm: " << xm << std::endl; 

    world.set_cell(Cells);

    /*check command line arguments for thread count*/
    int num_threads = std::thread::hardware_concurrency();

    std::cout<<"Running with "<<num_threads<<" threads"<<std::endl;
    world.setNumThreads(num_threads);   //set number of threads to use

    world.set_parameters(react_table_equation, react_type_table, react_prob_chemical, react_yield_p0, rn_coeffcients);
    // world.print_rn_matrix();
    world.print_rn_coeffcients();
    world.print_react_type_table();
    Species sp("test", 1, world);
    sp.inputParticle(particles);

    sp.printParticle(id);
    // sp.change_cell(5,5,5);
    sp.advance();
    // world.change_cell(5,5,5);
    // world.WprintCell(5,5,5);  


    std::cout << "After advance particle size: " << sp.getNp() <<  std::endl;

    sp.printParticle(id);

    int3 posInt = {(int)sp.particles[id].pos[0], (int)sp.particles[id].pos[1], (int)sp.particles[id].pos[2]};
    bool testIn = world.inFilm(posInt);

    if(testIn){
        std::cout << "Film in : " << std::endl;
        std::cout << "particle size: " << sp.particles.size() << std::endl;
        std::cout << "particles["<< id <<"].pos: " << sp.particles[id].pos << std::endl; 
    }
    else{
        std::cout << "Film out :" << std::endl;
        std::cout << "particle size: " << sp.particles.size() << std::endl;
        std::cout << "particles["<< id <<"].pos: " << sp.particles[id].pos << std::endl; 
    }


}






PYBIND11_MODULE(SimProfile, m) {
    // 绑定基础类型
    // PYBIND11_NUMPY_DTYPE(Cell, id, index, film, normal);

    bind_vec3<double>(m, "double3");
    bind_vec3<int>(m, "int3");
    
    // 绑定粒子类型
    bind_particle(m);
    // bind_Cell(m);
    
        // 绑定 Simulation 类
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<int, int, int, int>(), py::arg("seed"), py::arg("ni"),py::arg("nj"),py::arg("nk"))
        .def("add_particle", &Simulation::addParticle,
             py::arg("pos"), py::arg("vel"), py::arg("E"), py::arg("id"))
        .def("remove_particle", &Simulation::removeParticle,
             py::arg("id"))
        .def("get_particles", &Simulation::getParticles)
        .def("printParticle", &Simulation::printParticle)
        .def("moveParticle", &Simulation::moveParticle)
        .def("crossTest", &Simulation::crossTest)
        .def("getCells", &Simulation::getCells)
        .def("inputCell", &Simulation::inputCell, 
            py::arg("typeid"),
            py::arg("index"),
            py::arg("normal"),
            py::arg("film"), "typeid, index, normal, film")
        .def("printCell", &Simulation::printCell)
        .def("runSimulation", &Simulation::runSimulation)
        .def("normal_to_numpy", &Simulation::normal_to_numpy)
        .def("print_react_table_equation", &Simulation::print_react_table_equation)
        .def("set_all_parameters", &Simulation::set_all_parameters,
            py::arg("react_table_equation"),
            py::arg("react_type_table"),
            py::arg("react_prob_chemical"),
            py::arg("react_yield_p0"),
            py::arg("rn_coeffcients"), "react_table_equation, react_type_table, react_prob_chemical")
        .def("inputParticle", &Simulation::inputParticle, 
            py::arg("pos"),
            py::arg("vel"),
            py::arg("E"),
            py::arg("id"), "pos, vel, E, id");

}