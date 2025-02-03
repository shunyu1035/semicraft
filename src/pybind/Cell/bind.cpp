#include "film_update.h"
#include <pybind11/iostream.h>  // 用于重定向输出
#include <iostream>

// 基础打印函数
void print_values(int a, double b) {
    std::cout << "C++ 打印的值: a = " << a << ", b = " << b << std::endl;
}

// 重定向输出到 Python 的 sys.stdout
void redirect_output() {
    py::scoped_ostream_redirect stream(
        std::cout,
        py::module_::import("sys").attr("stdout")
    );
    std::cout << "此输出已重定向到 Python 的 sys.stdout" << std::endl;
}


PYBIND11_MODULE(film_optimized, m) {

    PYBIND11_NUMPY_DTYPE(Cell, id, index, film, normal);
    // 绑定 Cell 结构体
    // py::class_<Cell>(m, "Cell")
    //     .def(py::init<>())
    //     .def_readwrite("id", &Cell::id)
    //     .def_readwrite("index", &Cell::index)
    //     .def_readwrite("film", &Cell::film)
    //     .def_readwrite("normal", &Cell::normal);

    // 绑定核心函数
    m.def("update_film_label_index_normal_etch", 
        &update_film_label_index_normal_etch,
        "Optimized film update function",
        py::arg("cells"), 
        py::arg("point_etch"), 
        py::arg("cell_size_xyz"));

    m.def("print_values", &print_values, "打印整数和浮点数");

    m.def("redirect_output", &redirect_output, "重定向输出示例");
}
