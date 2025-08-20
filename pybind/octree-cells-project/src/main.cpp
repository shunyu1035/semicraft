#include "OctreeNode.h"
#include "Cell.h"
#include <iostream>
#include <functional>
#include <chrono>
#include "dda.h"

inline int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}


void printOctree(const OctreeNode* node, int depth = 0) {
    if (!node) return;
    std::string indent(depth * 2, ' ');
    std::cout << indent << "Node: ["
              << node->x0 << " " << node->x1 << ", "
              << node->y0 << " " << node->y1 << ", "
              << node->z0 << " " << node->z1 << "], "
              << "typeID: " << node->typeID << std::endl;
    for (const auto& child : node->children) {
        printOctree(child.get(), depth + 1);
    }
}

void printOctreeAtDepth(const OctreeNode* node, int targetDepth, int currentDepth = 0) {
    if (!node) return;
    if (currentDepth == targetDepth) {
        std::cout << "Node: [" 
                  << node->x0 << " " << node->x1 << ", "
                  << node->y0 << " " << node->y1 << ", "
                  << node->z0 << " " << node->z1 << "], "
                  << "typeID: " << node->typeID << std::endl;
        return;
    }
    for (const auto& child : node->children) {
        printOctreeAtDepth(child.get(), targetDepth, currentDepth + 1);
    }
}

void printType1Nodes(const OctreeNode* node, int depth = 0) {
    if (!node) return;
    std::string indent(depth * 2, ' ');
    std::cout << indent << "Node: ["
              << node->x0 << " " << node->x1 << ", "
              << node->y0 << " " << node->y1 << ", "
              << node->z0 << " " << node->z1 << "], "
              << "typeID: " << node->typeID << std::endl;
    if (node->typeID == 1) {
        for (const auto& child : node->children) {
            printType1Nodes(child.get(), depth + 1);
        }
    }
}

int main() {
    int nx = 32*4, ny = 32*4, nz = 32*4;
    std::vector<std::vector<std::vector<Cell>>> cells(
        nx, std::vector<std::vector<Cell>>(
            ny, std::vector<Cell>(
                nz, Cell(0, {0,0,0}, {0,0,0}, {}, 0.0)
            )
        )
    );
    // 设置某个cell为1
    // cells[120][3][4].typeID = 1;
    // cells[120][3][10].typeID = 1;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            cells[i][j][10].typeID = 1;
        }
    }

    int max_dim = std::max({nx, ny, nz});
    int tree_size = nextPowerOfTwo(max_dim);

    int maxBlockSize = 128;
    auto root = OctreeNode::buildMaskedOctree(cells, nx, ny, nz, maxBlockSize);
    // printOctree(root.get());
    
    // 打印最深层（叶节点）
    // int maxDepth = 0;
    // std::function<void(const OctreeNode*, int)> findMaxDepth = [&](const OctreeNode* node, int depth) {
    //     if (!node) return;
    //     if (node->children.empty()) maxDepth = std::max(maxDepth, depth);
    //     for (const auto& child : node->children)
    //         findMaxDepth(child.get(), depth + 1);
    // };
    // findMaxDepth(root.get(), 0);
    // printOctreeAtDepth(root.get(), maxDepth);

    // 打印第 n 层
    // int n = 4; // 这里可以设置想要打印的层数

    // std::cout << "Nodes at depth " << n << ":" << std::endl;
    // printOctreeAtDepth(root.get(), n);

    std::cout << "Root: [" << root->x0 << " " << root->x1 << ", "
              << root->y0 << " " << root->y1 << ", "
              << root->z0 << " " << root->z1 << "], "
              << "typeID: " << root->typeID << std::endl;

    for (size_t i = 0; i < root->children.size(); ++i) {
        auto& child = root->children[i];
        std::cout << "Child " << i << ": ["
                  << child->x0 << " " << child->x1 << ", "
                  << child->y0 << " " << child->y1 << ", "
                  << child->z0 << " " << child->z1 << "], "
                  << "typeID: " << child->typeID << std::endl;

        // 只展开typeID为1的节点
        if (child->typeID == 1) {
            for (size_t j = 0; j < child->children.size(); ++j) {
                auto& grandchild = child->children[j];
                std::cout << "  Grandchild " << j << ": ["
                          << grandchild->x0 << " " << grandchild->x1 << ", "
                          << grandchild->y0 << " " << grandchild->y1 << ", "
                          << grandchild->z0 << " " << grandchild->z1 << "], "
                          << "typeID: " << grandchild->typeID << std::endl;
            }
        }
    }


    // 打印所有typeID为1的节点
    printType1Nodes(root.get());


    Ray ray;
    ray.origin = {2.4, 3.2, 100.1};
    ray.direction = {0.5, 0.01, -0.02};

    jumpToTopPeriodic(nx, ny, nz, ray.origin, ray.direction, 12);

    std::cout << "Ray origin: [" 
              << ray.origin.x << ", " 
              << ray.origin.y << ", " 
              << ray.origin.z << "]\n";

    auto t1 = std::chrono::high_resolution_clock::now();
    traverseOctreeDDA(root.get(), ray);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Octree AABB traverse time: "
              << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";

    auto t3 = std::chrono::high_resolution_clock::now();
    voxelDDA(cells, nx, ny, nz, ray.origin, ray.direction);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Voxel DDA traverse time: "
              << std::chrono::duration<double, std::milli>(t4-t3).count() << " ms\n";

    // auto t5 = std::chrono::high_resolution_clock::now();
    // octreeDDA(root.get(), ray, nx, ny, nz);
    // auto t6 = std::chrono::high_resolution_clock::now();
    // std::cout << "Octree DDA traverse time: "
    //         << std::chrono::duration<double, std::milli>(t6-t5).count() << " ms\n";


    // auto t7 = std::chrono::high_resolution_clock::now();
    // octreeDDA_voxelstyle(root.get(), ray);
    // auto t8 = std::chrono::high_resolution_clock::now();
    // std::cout << "octreeDDA_voxelstyle traverse time: "
    //         << std::chrono::duration<double, std::milli>(t8-t7).count() << " ms\n";

    auto x_blocks = decomposeToPowersOfTwoLimited(nx);
    auto y_blocks = decomposeToPowersOfTwoLimited(ny);
    auto z_blocks = decomposeToPowersOfTwoLimited(nz);



    return 0;
}