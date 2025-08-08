#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "Field.h"



struct OctreeNode {
    int state; // 0: 空, 1: solid, -1: 混合
    OctreeNode* parent;
    std::vector<std::unique_ptr<OctreeNode>> children;

    OctreeNode(OctreeNode* parent = nullptr)
        : state(0), parent(parent), children(8, nullptr) {}
};


void setSolid(OctreeNode* node, int x0, int y0, int z0, int size,
              int x, int y, int z) {
    if (size == 1) {
        if (node->state == 1)
            return; // 已经是solid了，无需更新

        node->state = 1;

        // 优化的向上更新
        OctreeNode* current = node->parent;
        while (current != nullptr && current->state != 1) {
            current->state = 1;
            current = current->parent;
        }
        return;
    }

    int half = size / 2;
    int idx = ((x >= x0 + half) ? 1 : 0)
            | ((y >= y0 + half) ? 2 : 0)
            | ((z >= z0 + half) ? 4 : 0);

    int dx = (idx & 1) ? half : 0;
    int dy = (idx & 2) ? half : 0;
    int dz = (idx & 4) ? half : 0;

    if (!node->children[idx]) {
        node->children[idx] = std::make_unique<OctreeNode>(node);
    }

    if (node->state == 0)
        node->state = -1; // 从空变为混合

    setSolid(node->children[idx].get(), x0 + dx, y0 + dy, z0 + dz, half, x, y, z);
}



// ================== 插入 Solid ==================
void insertSolid(OctreeNode* node, int x0, int y0, int z0, int size,
                 int x, int y, int z) {
    if (size == 1) {
        if (node->state == 1)
            return;
        node->state = 1;
        OctreeNode* current = node->parent;
        while (current != nullptr && current->state != 1) {
            current->state = 1;
            current = current->parent;
        }
        return;
    }

    int half = size / 2;
    int idx = ((x >= x0 + half) ? 1 : 0)
            | ((y >= y0 + half) ? 2 : 0)
            | ((z >= z0 + half) ? 4 : 0);

    int dx = (idx & 1) ? half : 0;
    int dy = (idx & 2) ? half : 0;
    int dz = (idx & 4) ? half : 0;

    if (!node->children[idx]) {
        node->children[idx] = std::make_unique<OctreeNode>(node);
    }

    if (node->state == 0)
        node->state = -1;

    insertSolid(node->children[idx].get(), x0 + dx, y0 + dy, z0 + dz, half, x, y, z);
}

// ================== 删除 Solid ==================
bool tryCollapse(OctreeNode* node) {
    bool allEmpty = true;
    for (auto& child : node->children) {
        if (child && child->state != 0) {
            allEmpty = false;
            break;
        }
    }
    if (allEmpty) {
        node->children.clear();
        node->children.resize(8, nullptr);
        node->state = 0;
        return true;
    }
    return false;
}

void removeSolid(OctreeNode* node, int x0, int y0, int z0, int size,
                 int x, int y, int z) {
    if (size == 1) {
        node->state = 0;
        OctreeNode* current = node->parent;
        while (current != nullptr) {
            if (!tryCollapse(current))
                break;
            current = current->parent;
        }
        return;
    }

    int half = size / 2;
    int idx = ((x >= x0 + half) ? 1 : 0)
            | ((y >= y0 + half) ? 2 : 0)
            | ((z >= z0 + half) ? 4 : 0);

    int dx = (idx & 1) ? half : 0;
    int dy = (idx & 2) ? half : 0;
    int dz = (idx & 4) ? half : 0;

    if (!node->children[idx])
        return; // Already empty

    removeSolid(node->children[idx].get(), x0 + dx, y0 + dy, z0 + dz, half, x, y, z);
}

// ================== Ray Casting (Axis-aligned) ==================
bool rayIntersectsSolid(OctreeNode* node, int x0, int y0, int z0, int size,
                        float ox, float oy, float oz,
                        float dx, float dy, float dz,
                        float tMin, float tMax) {
    if (node->state == 0) return false;
    if (node->state == 1 || size == 1) return true;

    int half = size / 2;
    for (int i = 0; i < 8; ++i) {
        int dx_ = (i & 1) ? half : 0;
        int dy_ = (i & 2) ? half : 0;
        int dz_ = (i & 4) ? half : 0;

        int cx = x0 + dx_;
        int cy = y0 + dy_;
        int cz = z0 + dz_;

        if (!node->children[i]) continue;

        // AABB test
        float t0x = (cx - ox) / dx;
        float t1x = (cx + half - ox) / dx;
        float t0y = (cy - oy) / dy;
        float t1y = (cy + half - oy) / dy;
        float t0z = (cz - oz) / dz;
        float t1z = (cz + half - oz) / dz;

        float tmin = std::fmax(std::fmax(std::fmin(t0x, t1x), std::fmin(t0y, t1y)), std::fmin(t0z, t1z));
        float tmax = std::fmin(std::fmin(std::fmax(t0x, t1x), std::fmax(t0y, t1y)), std::fmax(t0z, t1z));

        if (tmax >= std::fmax(0.0f, tmin) && tmax >= tMin && tmin <= tMax) {
            if (rayIntersectsSolid(node->children[i].get(), cx, cy, cz, half,
                                   ox, oy, oz, dx, dy, dz, tMin, tMax)) {
                return true;
            }
        }
    }
    return false;
}

// ================== 示例 ==================
int main() {
    const int N = 256;
    auto root = std::make_unique<OctreeNode>();

    // 插入几个 solid voxel
    insertSolid(root.get(), 0, 0, 0, N, 100, 100, 100);
    insertSolid(root.get(), 0, 0, 0, N, 120, 120, 120);

    // 射线从原点出发，沿着对角线方向
    bool hit = rayIntersectsSolid(root.get(), 0, 0, 0, N,
                                   0, 0, 0, 1, 1, 1,
                                   0, 500);
    std::cout << "Ray hit: " << hit << std::endl;

    // 删除其中一个 voxel
    removeSolid(root.get(), 0, 0, 0, N, 100, 100, 100);

    // 再次射线测试
    hit = rayIntersectsSolid(root.get(), 0, 0, 0, N,
                              0, 0, 0, 1, 1, 1,
                              0, 500);
    std::cout << "Ray hit after deletion: " << hit << std::endl;

    return 0;
}
