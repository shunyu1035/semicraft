#ifndef INDEX_H
#define INDEX_H

#include <array>

struct int3 {
    int x, y, z;

    int3(int x = 0, int y = 0, int z = 0) : x(x), y(y), z(z) {}
};

struct double3 {
    double x, y, z;

    double3(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
};

struct BoundingBox {
    double3 min; // Minimum corner of the bounding box
    double3 max; // Maximum corner of the bounding box

    BoundingBox(const double3& min = double3(), const double3& max = double3()) : min(min), max(max) {}
};

#endif // INDEX_H