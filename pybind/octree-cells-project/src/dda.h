#include <cmath>
#include <iostream>
#include <stack>

// 返回true表示命中typeID==1的体素，返回false表示未命中
inline bool voxelDDA(const std::vector<std::vector<std::vector<Cell>>>& cells,
                     int nx, int ny, int nz,
                     double3 origin, double3 direction,
                     int maxSteps = 1000)
{
    // 起点
    int x = int(std::floor(origin.x));
    int y = int(std::floor(origin.y));
    int z = int(std::floor(origin.z));

    // 步进方向
    int stepX = (direction.x > 0) ? 1 : -1;
    int stepY = (direction.y > 0) ? 1 : -1;
    int stepZ = (direction.z > 0) ? 1 : -1;

    // 下一个格点边界的距离
    double tMaxX = ((stepX > 0 ? (x + 1) : x) - origin.x) / direction.x;
    double tMaxY = ((stepY > 0 ? (y + 1) : y) - origin.y) / direction.y;
    double tMaxZ = ((stepZ > 0 ? (z + 1) : z) - origin.z) / direction.z;

    if (direction.x == 0) tMaxX = 1e30;
    if (direction.y == 0) tMaxY = 1e30;
    if (direction.z == 0) tMaxZ = 1e30;

    // 步进到下一个格点边界的距离
    double tDeltaX = std::abs(1.0 / direction.x);
    double tDeltaY = std::abs(1.0 / direction.y);
    double tDeltaZ = std::abs(1.0 / direction.z);

    for (int step = 0; step < maxSteps; ++step) {
        // 检查是否越界
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
            return false;

        // 命中typeID==1的体素
        if (cells[x][y][z].typeID == 1) {
            std::cout << "DDA hit voxel: [" << x << ", " << y << ", " << z << "]\n";
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



// // 使粒子快速跳跃到顶边
// inline bool jumpToTop(const std::vector<std::vector<std::vector<Cell>>>& cells,
//                      int nx, int ny, int nz,
//                      double3& origin, double3 direction,
//                      int top)
// {
//     double projectionX = (direction.x / direction.z) * (origin.z - double(top));
//     double projectionY = (direction.y / direction.z) * (origin.z - double(top));

//     origin.x = projectionX - std::floor(projectionX / double(nx));
//     origin.y = projectionY - std::floor(projectionY / double(ny));
//     origin.z = top;

// }

inline bool jumpToTopPeriodic(int nx, int ny, int nz,
                              double3& origin, const double3& direction,
                              int top)
{
    if (std::abs(direction.z) < 1e-12) return false;
    double t = (double(top) - origin.z) / direction.z;
    if (t < 0) return false;

    double projectionX = origin.x + direction.x * t;
    double projectionY = origin.y + direction.y * t;

    // 周期性镜像空间
    origin.x = std::fmod(projectionX + nx, nx);
    origin.y = std::fmod(projectionY + ny, ny);
    origin.z = top;

    return true;
}


// inline bool pointInNode(const OctreeNode* node, const double3& pos) {
//     return pos.x >= node->x0 && pos.x < node->x1 &&
//            pos.y >= node->y0 && pos.y < node->y1 &&
//            pos.z >= node->z0 && pos.z < node->z1;
// }


// // 八叉树加速DDA
// inline bool octreeDDA(const OctreeNode* node, const Ray& ray, int nx, int ny, int nz, int maxSteps = 1000, int depth = 0)
// {
//     if (!node) return false;

//     double tmin, tmax;

//     double3 pos = ray.origin;
//     if (!pointInNode(node, pos)) return false;

//     // 步进起点
//     double t = std::max(0.0, tmin);

//     // 如果typeID==0，直接跳到tmax
//     if (node->typeID == 0) {

//         double nodeSize = node->x1 - node->x0;
//         // 起点
//         int x = int(std::floor(ray.origin.x));
//         int y = int(std::floor(ray.origin.y));
//         int z = int(std::floor(ray.origin.z));
//         // 步进方向
//         int stepX = (ray.direction.x > 0) ? nodeSize : -nodeSize;
//         int stepY = (ray.direction.y > 0) ? nodeSize : -nodeSize;
//         int stepZ = (ray.direction.z > 0) ? nodeSize : -nodeSize;

//         // 下一个格点边界的距离
//         double tMaxX = ((stepX > 0 ? (x + nodeSize) : x) - ray.origin.x) / ray.direction.x;
//         double tMaxY = ((stepY > 0 ? (y + nodeSize) : y) - ray.origin.y) / ray.direction.y;
//         double tMaxZ = ((stepZ > 0 ? (z + nodeSize) : z) - ray.origin.z) / ray.direction.z;

//         if (ray.direction.x == 0) tMaxX = 1e30;
//         if (ray.direction.y == 0) tMaxY = 1e30;
//         if (ray.direction.z == 0) tMaxZ = 1e30;

//         // 步进到下一个格点边界的距离
//         double tDeltaX = std::abs(nodeSize / ray.direction.x);
//         double tDeltaY = std::abs(nodeSize / ray.direction.y);
//         double tDeltaZ = std::abs(nodeSize / ray.direction.z);

//         for (int step = 0; step < maxSteps; ++step) {
//             // 检查是否越界
//             if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
//                 return false;

//             // 命中typeID==1的体素
//             // if (cells[x][y][z].typeID == 1) {
//             //     std::cout << "DDA hit voxel: [" << x << ", " << y << ", " << z << "]\n";
//             //     return true;
//             // }

//             // 步进到下一个格点
//             if (tMaxX < tMaxY && tMaxX < tMaxZ) {
//                 x += stepX;
//                 tMaxX += tDeltaX;
//             } else if (tMaxY < tMaxZ) {
//                 y += stepY;
//                 tMaxY += tDeltaY;
//             } else {
//                 z += stepZ;
//                 tMaxZ += tDeltaZ;
//             }
//         }

//         std::cout << "octreeDDA skip empty node: [" << node->x0 << "," << node->x1 << "," << node->y0 << "," << node->y1 << "," << node->z0 << "," << node->z1 << "]\n";
//         // 跳到tmax继续DDA（可在父节点或上层循环实现）
//         // 这里直接return false，实际应用中应由上层循环继续
//         return false;
//     }

//     // 叶节点，逐体素DDA
//     if (node->children.empty()) {
//         int x = int(std::floor(pos.x));
//         int y = int(std::floor(pos.y));
//         int z = int(std::floor(pos.z));
//         for (int step = 0; step < maxSteps; ++step) {
//             if (x < node->x0 || x >= node->x1 || y < node->y0 || y >= node->y1 || z < node->z0 || z >= node->z1)
//                 return false;
//             if (node->typeID == 1) {
//                 std::cout << "octreeDDA hit leaf voxel: [" << x << ", " << y << ", " << z << "]\n";
//                 return true;
//             }
//             // 步进到下一个体素
//             double tx = ((ray.direction.x > 0 ? (x + 1) : x) - pos.x) / ray.direction.x;
//             double ty = ((ray.direction.y > 0 ? (y + 1) : y) - pos.y) / ray.direction.y;
//             double tz = ((ray.direction.z > 0 ? (z + 1) : z) - pos.z) / ray.direction.z;
//             if (ray.direction.x == 0) tx = 1e30;
//             if (ray.direction.y == 0) ty = 1e30;
//             if (ray.direction.z == 0) tz = 1e30;
//             if (tx < ty && tx < tz) { x += (ray.direction.x > 0 ? 1 : -1); pos.x += (ray.direction.x > 0 ? 1 : -1); }
//             else if (ty < tz) { y += (ray.direction.y > 0 ? 1 : -1); pos.y += (ray.direction.y > 0 ? 1 : -1); }
//             else { z += (ray.direction.z > 0 ? 1 : -1); pos.z += (ray.direction.z > 0 ? 1 : -1); }
//         }
//         return false;
//     }

//     // typeID==1且有子节点，递归进入子节点
//     for (const auto& child : node->children) {
//         if (octreeDDA(child.get(), ray, nx, ny, nz, maxSteps, depth + 1))
//             return true;
//     }
//     return false;
// }

// // 非递归八叉树DDA
// inline bool octreeDDA_iter(const OctreeNode* root, const Ray& ray, int nx, int ny, int nz, int maxSteps = 10000)
// {
//     struct StackNode {
//         const OctreeNode* node;
//         double tmin, tmax;
//     };

//     std::stack<StackNode> stack;
//     double tmin_root, tmax_root;
//     if (!rayIntersectsAABB(ray, root, tmin_root, tmax_root)) return false;
//     stack.push({root, tmin_root, tmax_root});

//     while (!stack.empty()) {
//         auto [node, tmin, tmax] = stack.top();
//         stack.pop();

//         if (!node) continue;

//         // 跳过空节点
//         if (node->typeID == 0) {
//             // std::cout << "Skip empty node: [" << node->x0 << "," << node->x1 << "," << node->y0 << "," << node->y1 << "," << node->z0 << "," << node->z1 << "]\n";
//             continue;
//         }

//         // 叶节点，体素DDA
//         if (node->children.empty()) {
//             // 体素DDA步进
//             double t = std::max(0.0, tmin);
//             double3 pos = {
//                 ray.origin.x + ray.direction.x * t,
//                 ray.origin.y + ray.direction.y * t,
//                 ray.origin.z + ray.direction.z * t
//             };
//             int x = int(std::floor(pos.x));
//             int y = int(std::floor(pos.y));
//             int z = int(std::floor(pos.z));
//             for (int step = 0; step < maxSteps; ++step) {
//                 if (x < node->x0 || x >= node->x1 || y < node->y0 || y >= node->y1 || z < node->z0 || z >= node->z1)
//                     break;
//                 if (node->typeID == 1) {
//                     std::cout << "octreeDDA_iter hit leaf voxel: [" << x << ", " << y << ", " << z << "]\n";
//                     return true;
//                 }
//                 // 步进到下一个体素
//                 double tx = ((ray.direction.x > 0 ? (x + 1) : x) - pos.x) / ray.direction.x;
//                 double ty = ((ray.direction.y > 0 ? (y + 1) : y) - pos.y) / ray.direction.y;
//                 double tz = ((ray.direction.z > 0 ? (z + 1) : z) - pos.z) / ray.direction.z;
//                 if (ray.direction.x == 0) tx = 1e30;
//                 if (ray.direction.y == 0) ty = 1e30;
//                 if (ray.direction.z == 0) tz = 1e30;
//                 if (tx < ty && tx < tz) { x += (ray.direction.x > 0 ? 1 : -1); pos.x += (ray.direction.x > 0 ? 1 : -1); }
//                 else if (ty < tz) { y += (ray.direction.y > 0 ? 1 : -1); pos.y += (ray.direction.y > 0 ? 1 : -1); }
//                 else { z += (ray.direction.z > 0 ? 1 : -1); pos.z += (ray.direction.z > 0 ? 1 : -1); }
//             }
//             continue;
//         }

//         // typeID==1且有子节点，遍历所有与射线相交的子节点
//         for (const auto& child : node->children) {
//             double tmin_c, tmax_c;
//             if (rayIntersectsAABB(ray, child.get(), tmin_c, tmax_c)) {
//                 // 只推进到与射线相交的子节点
//                 stack.push({child.get(), tmin_c, tmax_c});
//             }
//         }
//     }
//     return false;
// }

// inline bool pointInNode(const OctreeNode* node, const double3& pos) {
//     return pos.x >= node->x0 && pos.x < node->x1 &&
//            pos.y >= node->y0 && pos.y < node->y1 &&
//            pos.z >= node->z0 && pos.z < node->z1;
// }

// inline bool octreeDDA_voxelstyle(const OctreeNode* root, const Ray& ray, int maxSteps = 10000)
// {
//     double3 pos = ray.origin;
//     const OctreeNode* node = root;
//     int steps = 0;

//     while (steps < maxSteps) {
//         // 1. 判断当前位置是否在node包围盒内
//         if (!pointInNode(node, pos)) return false;

//         // 2. 如果是叶节点，体素DDA
//         if (node->children.empty()) {
//             int x = int(std::floor(pos.x));
//             int y = int(std::floor(pos.y));
//             int z = int(std::floor(pos.z));
//             if (x >= node->x0 && x < node->x1 &&
//                 y >= node->y0 && y < node->y1 &&
//                 z >= node->z0 && z < node->z1 &&
//                 node->typeID == 1) {
//                 std::cout << "octreeDDA_voxelstyle hit leaf voxel: [" << x << ", " << y << ", " << z << "]\n";
//                 return true;
//             }
//             // 步进到下一个体素
//             double tx = ((ray.direction.x > 0 ? (x + 1) : x) - pos.x) / ray.direction.x;
//             double ty = ((ray.direction.y > 0 ? (y + 1) : y) - pos.y) / ray.direction.y;
//             double tz = ((ray.direction.z > 0 ? (z + 1) : z) - pos.z) / ray.direction.z;
//             if (ray.direction.x == 0) tx = 1e30;
//             if (ray.direction.y == 0) ty = 1e30;
//             if (ray.direction.z == 0) tz = 1e30;
//             double tstep = std::min({tx, ty, tz});
//             pos.x += ray.direction.x * tstep;
//             pos.y += ray.direction.y * tstep;
//             pos.z += ray.direction.z * tstep;
//             ++steps;
//             continue;
//         }

//         // 3. 如果typeID==0，直接步进到node边界
//         if (node->typeID == 0) {
//             // 步进到node的下一个边界
//             double tx = ((ray.direction.x > 0 ? node->x1 : node->x0) - pos.x) / ray.direction.x;
//             double ty = ((ray.direction.y > 0 ? node->y1 : node->y0) - pos.y) / ray.direction.y;
//             double tz = ((ray.direction.z > 0 ? node->z1 : node->z0) - pos.z) / ray.direction.z;
//             if (ray.direction.x == 0) tx = 1e30;
//             if (ray.direction.y == 0) ty = 1e30;
//             if (ray.direction.z == 0) tz = 1e30;
//             double tstep = std::min({tx, ty, tz});
//             pos.x += ray.direction.x * tstep;
//             pos.y += ray.direction.y * tstep;
//             pos.z += ray.direction.z * tstep;
//             ++steps;
//             // 重新从root开始查找包含pos的node
//             node = root;
//             continue;
//         }

//         // 4. typeID==1且有子节点，找到包含pos的子节点
//         bool found = false;
//         for (const auto& child : node->children) {
//             if (pointInNode(child.get(), pos)) {
//                 node = child.get();
//                 found = true;
//                 break;
//             }
//         }
//         if (!found) return false; // 没有包含pos的子节点，说明射线出界
//     }
//     return false;
// }