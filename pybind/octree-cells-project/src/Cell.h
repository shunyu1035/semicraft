// This file defines the Cell class, which represents a single cell in the three-dimensional space.

#ifndef CELL_H
#define CELL_H

#include <vector>
#include <mutex>


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


class Cell {
public:
    int typeID;
    int3 index; // Assuming int3 is defined in index.h
    double3 normal; // Assuming double3 is defined in index.h
    std::vector<int> film;
    double potential;
    std::mutex film_mutex; // Mutex for thread safety

    // Custom copy constructor
    Cell(const Cell& other) 
        : typeID(other.typeID), index(other.index), normal(other.normal), film(other.film), potential(other.potential) {}

    // Other constructors
    Cell(int typeID, int3 index, double3 normal, std::vector<int> film, double potential) 
        : typeID(typeID), index(index), normal(normal), film(film), potential(potential) {}
};

#endif // CELL_H