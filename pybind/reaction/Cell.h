#include <vector>
#include <array>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

// 定义 Cell 结构体 (与 Cython 对齐)
struct Cell {
    int id;
    std::array<int, 3> index;
    std::array<int, 5> film;
    std::array<double, 3> normal;
};




/*defines the computational domain*/
class World
{
public:	
	/*constructor, allocates memory*/
	World(int ni, int nj, int nk): 
     ni(ni), nj(nj), nk(nk), xm({(double)ni,(double)nj,(double)nk}) {}

	// /*functions to set mesh origin and spacing*/
	// void setExtents() {xm[0] = <double>ni; xm[1] = <double>nj; xm[2] = <double>nk;}
	
	double3 getXm() const {return double3(xm);}

	bool inBounds(double3 pos) {
		for (int i=0;i<3;i++)
			if (pos[i]< 0 || pos[i]>=xm[i]) return false;
		return true;
	}

    void print_cell(){
        int dim_0 = Cells.size();
        int dim_1 = Cells[0].size();
        int dim_2 = Cells[0][0].size();
        std::cout << "World Cell: " << dim_0 << " " << dim_1 << " " << dim_2 <<  std::endl;
    }
    void set_cell(std::vector<std::vector<std::vector<Cell>>> Cell){
        Cells = Cell;
        print_cell();
    }

	/*another form that takes 3 ints as inputs*/
	double3 pos(int i, int j, int k) {
		double3 x{(double)i,(double)j,(double)k};
		return x;
	}

	//mesh geometry
	const int ni,nj,nk;	//number of nodes

protected:
    std::vector<std::vector<std::vector<Cell>>> Cells;
	double3 xm;	//origin-diagonally opposite corner (max bound)

};













// 定义网格偏移常量
constexpr std::array<std::array<int, 3>, 6> GRID_CROSS = {{
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
    {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
}};

// 核心计算函数
py::array_t<int> update_film_label_index_normal_etch(
    py::array_t<Cell, py::array::c_style> cell_array,
    py::array_t<int, py::array::c_style> point_etch,
    py::array_t<int, py::array::c_style> cellSizeXYZ
) {
    // 获取 NumPy 数组的原始指针
    auto cell_buf = cell_array.request();
    auto* cell_ptr = static_cast<Cell*>(cell_buf.ptr);
    auto point_buf = point_etch.request();
    auto* point_ptr = static_cast<int*>(point_buf.ptr);
    
    const int num_points = point_etch.shape(0);
    const int grid_x = cell_array.shape(0);
    const int grid_y = cell_array.shape(1);
    const int grid_z = cell_array.shape(2);

    // 预分配输出数组
    py::array_t<int> result({num_points, 6, 3});
    auto res_buf = result.request();
    int* res_ptr = static_cast<int*>(res_buf.ptr);

    // OpenMP 并行优化
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        const int x = point_ptr[i * 3];
        const int y = point_ptr[i * 3 + 1];
        const int z = point_ptr[i * 3 + 2];

        // 标记当前 Cell
        cell_ptr[x * grid_y * grid_z + y * grid_z + z].id = -1;

        // 处理 6 个相邻方向
        for (int j = 0; j < 6; ++j) {
            const int nx = x + GRID_CROSS[j][0];
            const int ny = y + GRID_CROSS[j][1];
            const int nz = z + GRID_CROSS[j][2];

            // 边界检查
            if (nx >= 0 && nx < grid_x && 
                ny >= 0 && ny < grid_y && 
                nz >= 0 && nz < grid_z) {

                // 记录邻接点坐标
                res_ptr[i * 18 + j * 3] = nx;
                res_ptr[i * 18 + j * 3 + 1] = ny;
                res_ptr[i * 18 + j * 3 + 2] = nz;

                Cell& neighbor = cell_ptr[nx * grid_y * grid_z + ny * grid_z + nz];
                
                // 状态更新逻辑
                if (neighbor.id == 2) {
                    neighbor.id = 1;
                    // 处理下层邻接点
                    for (int m = 0; m < 6; ++m) {
                        const int mx = nx + GRID_CROSS[m][0];
                        const int my = ny + GRID_CROSS[m][1];
                        const int mz = nz + GRID_CROSS[m][2];
                        if (mx >= 0 && mx < grid_x && 
                            my >= 0 && my < grid_y && 
                            mz >= 0 && mz < grid_z) {
                            Cell& sub_neighbor = cell_ptr[mx * grid_y * grid_z + my * grid_z + mz];
                            if (sub_neighbor.id == 3) sub_neighbor.id = 2;
                        }
                    }
                } else if (neighbor.id == -1) {
                    // 处理真空邻接点
                    for (int l = 0; l < 6; ++l) {
                        const int lx = nx + GRID_CROSS[l][0];
                        const int ly = ny + GRID_CROSS[l][1];
                        const int lz = nz + GRID_CROSS[l][2];
                        if (lx >= 0 && lx < grid_x && 
                            ly >= 0 && ly < grid_y && 
                            lz >= 0 && lz < grid_z) {
                            Cell& vacuum_neighbor = cell_ptr[lx * grid_y * grid_z + ly * grid_z + lz];
                            if (vacuum_neighbor.id == 1) neighbor.id = 0;
                        }
                    }
                }
            }
        }
    }

    return result;
}