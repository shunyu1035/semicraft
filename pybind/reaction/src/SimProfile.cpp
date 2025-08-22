#include "SimProfile.h"


bool Simulation::need_recompute = false;

int Simulation::runSimulation(int time, int ArgonID, int depo_or_etch, bool redepo,
    bool diffusion, double diffusion_coeffient, int diffusion_distant, int stopPointY, int stopPointZ, double chemical_angle_v1, double chemical_angle_v2){
    // 注册信号处理器
    // std::signal(SIGSEGV, signalHandler);
    std::signal(SIGSEGV, globalSignalHandler);


    World world(ni, nj, nk, FILMSIZE, FilmDensity, ArgonID, redepo, diffusion, diffusion_coeffient, diffusion_distant,
        chemical_angle_v1, chemical_angle_v2);
    // world.print_rn_angle();

    std::cout << "grid_cross: " << std::endl; 
    for(size_t i=0; i<6; ++i){
        std::cout << world.grid_cross[i] << '\n'; 
    }


    double3 xm = world.getXm();
    std::cout << "World xm: " << xm << std::endl; 

    world.set_cell(typeID_in, potential_in, index_in, normal_in, film_in);
    world.sputter_yield_angle(sputter_yield_coefficient);
    // world.sputter_yield_angle(sputter_yield_coefficient[0], sputter_yield_coefficient[1], sputter_yield_coefficient[2]);
    world.inputParticle(particles);

    /*check command line arguments for thread count*/
    int num_threads = std::thread::hardware_concurrency();

    std::cout<<"Running with "<<num_threads<<" threads"<<std::endl;
    world.setNumThreads(num_threads);   //set number of threads to use

    world.set_parameters(react_table_equation, reflect_probability, reflect_coefficient, react_type_table, react_prob_chemical, react_yield_p0, film_eth, rn_coeffcients, E_decrease);
    // world.print_rn_matrix();
    // world.print_rn_coeffcients();
    // world.print_react_type_table();
    world.print_reflect_probability();
    world.print_react_yield_p0();
    Species sp("test", 1, world, max_particles);
    sp.inputParticle(particles);

    auto t_start = std::chrono::high_resolution_clock::now();

    int reaction_count = 0;
    // sp.change_cell(5,5,5);
    try {
        for(int t=0; t<time; ++t){
            // int reaction_count = 0;
            if (t % 5000 == 0) {  // 只有当 t 是 1000 的整数倍时才打印
                int film_thick = world.scan_bottom();
                std::cout << "Running " << t << " step; " << "thickness: " << film_thick << "; react_particles_count: " << reaction_count << std::endl;
            }
            // std::cout<<"Running "<< t <<" step; "  <<std::endl;
            sp.advance(std::ref(reaction_count));

            // int ret = sp.advance_DDA(std::ref(reaction_count), depo_or_etch, stopPointY, stopPointZ);
            // if (ret == 0) {
            //     break;
            // }
            // if(world.scan_bottom()){
            //     std::cout << "etching reach the bottom;" << std::endl;
            //     break;
            // }

            if(depo_or_etch == -1){
                if(world.scan_stopPoint(stopPointY, stopPointZ)){
                    std::cout << "etching reach the bottom;" << std::endl;
                    std::cout << "Total: Running " << t << " step; "  << "; react_particles_count: " << reaction_count  << std::endl;
                    break;
                }
            }
            else if(depo_or_etch == 1){
                if(world.scan_stopPoint_depo(stopPointY, stopPointZ)){
                    std::cout << "depo reach the top;" << std::endl;
                    std::cout << "Total: Running " << t << " step; " << "; react_particles_count: " << reaction_count  << std::endl;
                    break;
                }
            }
            // 检查错误标志，如果检测到错误，则抛出异常
            if (error_flag==1) {
                throw SimulationError("Simulation encountered a segmentation fault.");
                std::exit(1);
                break;
            }
        }
    } catch (const SimulationError& e) {
        std::cout << e.what() << std::endl;
        // 在捕获异常时跳出循环并返回 0
        return 1;
    }
    // sp.advance(reaction_count);
    // world.change_cell(5,5,5);
    // world.WprintCell(5,5,5);  
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Simulation main loop elapsed time: " << elapsed << " seconds" << std::endl;


    std::cout << "After advance particle size: " << sp.getNp() <<  std::endl;

    // sp.printParticle(id);

    // int3 posInt = {(int)sp.particles[id].pos[0], (int)sp.particles[id].pos[1], (int)sp.particles[id].pos[2]};
    // bool testIn = world.inFilm(posInt);
    // for(size_t i=0; i<particles.size(); ++i){
    // if(world.inBounds(sp.particles[i].pos)){        
    //     continue;
    // }
    // else{
        
    //     std::cout << "Film out :" << i << std::endl;
    //     std::cout << "particle size: " << sp.particles.size() << std::endl;
    //     std::cout << "particles["<< i <<"].pos: " << sp.particles[i].pos << std::endl; 
    //     std::cout << "particles["<< i <<"].vel: " << sp.particles[i].vel << std::endl; 
    //     std::cout << "particles["<< i <<"].id: " << sp.particles[i].id << std::endl; 
    //     std::cout << "particles["<< i <<"].E: " << sp.particles[i].E << std::endl; 
    //     std::cout << "Simulation particle size: " << particles.size() <<  std::endl;
    //     sp.showParticleIn(i);

    //     break;
    // }
    // }

    std::cout << "Film in :" << std::endl;
    // Cells = world.Cells;
    normal_in = world.output_normal_in();
    typeID_in = world.output_typeID_in();
    potential_in = world.output_potential_in();
    film_in = world.output_film_in();

    world.print_react_prob_chemical();

    return 0;
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
        .def(py::init<int, int, int, int, int, int, size_t>(), py::arg("seed"), py::arg("ni"),py::arg("nj"),py::arg("nk"),py::arg("FILMSIZE"),py::arg("FilmDensity"),py::arg("max_particles"))
        .def("add_particle", &Simulation::addParticle,
             py::arg("pos"), py::arg("vel"), py::arg("E"), py::arg("id"))
        .def("remove_particle", &Simulation::removeParticle,
             py::arg("id"))
        .def("get_particles", &Simulation::getParticles)
        .def("printParticle", &Simulation::printParticle)
        .def("moveParticle", &Simulation::moveParticle)
        .def("crossTest", &Simulation::crossTest)
        .def("inputCell", &Simulation::inputCell, 
            py::arg("typeid"),
            py::arg("potential"),
            py::arg("index"),
            py::arg("normal"),
            py::arg("film"), "typeid, potential, index, normal, film")
        .def("runSimulation", &Simulation::runSimulation)
        .def("normal_to_numpy", &Simulation::normal_to_numpy)
        .def("cell_data_to_numpy", &Simulation::cell_data_to_numpy)
        .def("print_react_table_equation", &Simulation::print_react_table_equation)
        .def("set_all_parameters", &Simulation::set_all_parameters,
            py::arg("react_table_equation"),
            py::arg("react_type_table"),
            py::arg("reflect_probability"),
            py::arg("reflect_coefficient"),
            py::arg("react_prob_chemical"),
            py::arg("react_yield_p0"),
            py::arg("film_eth"),
            py::arg("rn_coeffcients"),
            py::arg("E_decrease"), "react_table_equation, react_type_table, react_prob_chemical")
        .def("inputParticle", &Simulation::inputParticle, 
            py::arg("pos"),
            py::arg("vel"),
            py::arg("E"),
            py::arg("id"), "pos, vel, E, id")
        .def("input_sputter_yield_coefficient", &Simulation::input_sputter_yield_coefficient)
        .def("recompute", &Simulation::recompute, "A function that returns True");

    // 可选：将 SimulationError 也绑定为 Python 异常
    static py::exception<SimulationError> ex(m, "SimulationError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const SimulationError &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

}