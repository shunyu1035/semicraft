#include <cmath>
#include <iostream>
#include <stack>

// 返回true表示命中typeID==1的体素，返回false表示未命中
inline bool voxelDDA(const std::vector<std::vector<std::vector<Cell>>>& cells,
                     int nx, int ny, int nz,
                     double3& origin, double3 direction,
                     int maxSteps = 1000)
{
    // 起点
    int x = int(std::floor(origin[0]));
    int y = int(std::floor(origin[1]));
    int z = int(std::floor(origin[2]));

    // 步进方向
    int stepX = (direction[0] > 0) ? 1 : -1;
    int stepY = (direction[1] > 0) ? 1 : -1;
    int stepZ = (direction[2] > 0) ? 1 : -1;

    // 下一个格点边界的距离
    double tMaxX = ((stepX > 0 ? (x + 1) : x) - origin[0]) / direction[0];
    double tMaxY = ((stepY > 0 ? (y + 1) : y) - origin[1]) / direction[1];
    double tMaxZ = ((stepZ > 0 ? (z + 1) : z) - origin[2]) / direction[2];

    if (direction[0] == 0) tMaxX = 1e30;
    if (direction[1] == 0) tMaxY = 1e30;
    if (direction[2] == 0) tMaxZ = 1e30;

    // 步进到下一个格点边界的距离
    double tDeltaX = std::abs(1.0 / direction[0]);
    double tDeltaY = std::abs(1.0 / direction[1]);
    double tDeltaZ = std::abs(1.0 / direction[2]);

    for (int step = 0; step < maxSteps; ++step) {
        // 检查是否越界
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
            return false;

        // 命中typeID==1的体素
        if (cells[x][y][z].typeID == 1) {
            // std::cout << "DDA hit voxel: [" << x << ", " << y << ", " << z << "]\n";

            origin[0] = x;
            origin[1] = y;
            origin[2] = z;
            return true;
        }

        // 步进到下一个格点
        if (tMaxX < tMaxY && tMaxX < tMaxZ) {
            x += stepX;
            tMaxX += tDeltaX;
        } else if (tMaxY < tMaxZ) {
            y += stepY;
            tMaxY += tDeltaY;
        } else {
            z += stepZ;
            tMaxZ += tDeltaZ;
        }
    }
    return false;
}



inline bool jumpToTopPeriodic(int nx, int ny, int nz,
                              double3& origin, double3& direction,
                              int top)
{
    if (std::abs(direction[2]) < 1e-12) return false;
    double t = (double(top) - origin[2]) / direction[2];
    if (t < 0) return false;

    double projectionX = origin[0] + direction[0] * t;
    double projectionY = origin[1] + direction[1] * t;

    // 周期性镜像空间，保证结果在 [0, nx) 和 [0, ny)
    auto mod = [](double x, double n) {
        double r = std::fmod(x, n);
        return (r < 0) ? r + n : r;
    };
    origin[0] = mod(projectionX, nx);
    origin[1] = mod(projectionY, ny);
    origin[2] = top;

    // std::cout << "origin after jump: (" 
    //           << origin[0] << ", " 
    //           << origin[1] << ", " 
    //           << origin[2] << ")" << std::endl;
    return true;
}

