#ifndef OCTREENODE_H
#define OCTREENODE_H

#include "Cell.h"
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <algorithm>


// 将n分解为若干2的幂之和（从大到小）
inline std::vector<int> decomposeToPowersOfTwo(int n) {
    std::vector<int> result;
    for (int p = 1 << 30; p > 0; p >>= 1) {
        if (n >= p) {
            result.push_back(p);
            n -= p;
        }
    }
    return result;
}

inline int maxPowerOfTwoLE(int n) {
    int p = 1;
    while (p << 1 <= n) p <<= 1;
    return p;
}

inline std::vector<int> decomposeToPowersOfTwoLimited(int n) {
    std::vector<int> result;
    while (n > 0) {
        int p = maxPowerOfTwoLE(n);
        result.push_back(p);
        n -= p;
    }
    return result;
}

inline std::vector<int> decomposeToPowersOfTwoLimitedMax(int n, int maxBlockSize) {
    std::vector<int> result;
    while (n > 0) {
        int p = 1;
        while ((p << 1) <= n && (p << 1) <= maxBlockSize) p <<= 1;
        if (p > n) p = n; // 最后一个块可能不是2的幂
        result.push_back(p);
        n -= p;
    }
    return result;
}

inline std::vector<std::pair<int, int>> splitToPowersOfTwoBlocks(int n) {
    std::vector<std::pair<int, int>> blocks;
    int start = 0;
    while (n > 0) {
        int p = 1;
        while ((p << 1) <= n) p <<= 1;
        blocks.emplace_back(start, start + p);
        start += p;
        n -= p;
    }
    return blocks;
}

inline std::vector<std::pair<int, int>> splitToPowersOfTwoBlocksMax(int n, int maxBlockSize) {
    std::vector<std::pair<int, int>> blocks;
    int start = 0;
    for (int len : decomposeToPowersOfTwoLimitedMax(n, maxBlockSize)) {
        blocks.emplace_back(start, start + len);
        start += len;
    }
    return blocks;
}

struct OctreeNode {
    int typeID = 0;
    int x0, x1, y0, y1, z0, z1;
    std::vector<std::unique_ptr<OctreeNode>> children;

    OctreeNode(int x0, int x1, int y0, int y1, int z0, int z1)
        : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1) {}

    // 区块内部递归八叉树（区块size为2的幂）
    static std::unique_ptr<OctreeNode> buildBlockOctree(
        const std::vector<std::vector<std::vector<Cell>>>& cells,
        int x0, int y0, int z0, int size, int nx, int ny, int nz)
    {
        auto node = std::make_unique<OctreeNode>(x0, x0+size, y0, y0+size, z0, z0+size);
        if (size == 1) {
            if (x0 < nx && y0 < ny && z0 < nz)
                node->typeID = cells[x0][y0][z0].typeID;
            else
                node->typeID = 0;
            return node;
        }
        bool allZero = true;
        int half = size / 2;
        for (int oct = 0; oct < 8; ++oct) {
            int dx = (oct & 1) ? half : 0;
            int dy = (oct & 2) ? half : 0;
            int dz = (oct & 4) ? half : 0;
            auto child = buildBlockOctree(
                cells, x0 + dx, y0 + dy, z0 + dz, half, nx, ny, nz);
            if (child->typeID == 1) allZero = false;
            node->children.push_back(std::move(child));
        }
        node->typeID = allZero ? 0 : 1;
        return node;
    }
 
    // 第一层mask分块
    static std::unique_ptr<OctreeNode> buildMaskedOctree(
        const std::vector<std::vector<std::vector<Cell>>>& cells,
        int nx, int ny, int nz, int maxBlockSize)
    {
        auto x_blocks = splitToPowersOfTwoBlocksMax(nx, maxBlockSize);
        auto y_blocks = splitToPowersOfTwoBlocksMax(ny, maxBlockSize);
        auto z_blocks = splitToPowersOfTwoBlocksMax(nz, maxBlockSize);

        auto root = std::make_unique<OctreeNode>(0, nx, 0, ny, 0, nz);

        for (const auto& xb : x_blocks)
            for (const auto& yb : y_blocks)
                for (const auto& zb : z_blocks) {
                    int x0 = xb.first, x1 = xb.second;
                    int y0 = yb.first, y1 = yb.second;
                    int z0 = zb.first, z1 = zb.second;
                    int block_size = std::max({x1-x0, y1-y0, z1-z0});
                    auto child = OctreeNode::buildBlockOctree(
                        cells, x0, y0, z0, block_size, nx, ny, nz);
                    root->children.push_back(std::move(child));
                }
        root->typeID = 0;
        for (const auto& child : root->children)
            if (child->typeID == 1) root->typeID = 1;
        return root;
    }
};

struct Ray {
    double3 origin;
    double3 direction;
};

inline bool rayIntersectsAABB(const Ray& ray, const OctreeNode* node, double tmin = 0.0, double tmax = 1e9) {
    double3 bounds_min = {double(node->x0), double(node->y0), double(node->z0)};
    double3 bounds_max = {double(node->x1), double(node->y1), double(node->z1)};
    for (int i = 0; i < 3; ++i) {
        double origin = (&ray.origin.x)[i];
        double dir = (&ray.direction.x)[i];
        double minb = (&bounds_min.x)[i];
        double maxb = (&bounds_max.x)[i];
        // std::cout << "origin: " << origin  << " dir: " << dir  << " minb: " << minb << " maxb: " << maxb << std::endl;
        if (std::abs(dir) < 1e-12) {
            if (origin < minb || origin > maxb) {
                // std::cout << "out " <<  std::endl;
                return false;
            }
            // std::cout << "out " <<  std::endl; 
        } else {
            double t1 = (minb - origin) / dir;
            double t2 = (maxb - origin) / dir;

            // std::cout << "t1: " << t1  << " t2: " << t2  << std::endl;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);

            // std::cout << "tmin: " << tmin  << " tmax: " << tmax  << std::endl;
            if (tmin > tmax) return false;
        }
    }
    return true;
}
 
inline void traverseOctreeDDA(const OctreeNode* node, const Ray& ray) {
    if (!node) return;
    if (!rayIntersectsAABB(ray, node)) return;
    if (node->typeID == 0) {
        // std::cout << "Skip empty node: ["
        //           << node->x0 << " " << node->x1 << ", "
        //           << node->y0 << " " << node->y1 << ", "
        //           << node->z0 << " " << node->z1 << "], "
        //           << "typeID: " << node->typeID << std::endl;
        return; // 空区块直接跳过
    }

    if (node->children.empty()) {
        // 到达叶节点，做格点碰撞检测
        std::cout << "Hit leaf node: [" 
                  << node->x0 << " " << node->x1 << ", "
                  << node->y0 << " " << node->y1 << ", "
                  << node->z0 << " " << node->z1 << "], "
                  << "typeID: " << node->typeID << std::endl;
        return;
    }
    // 递归遍历所有子节点
    for (const auto& child : node->children) {
        traverseOctreeDDA(child.get(), ray);
    }
}



#endif // OCTREENODE_H