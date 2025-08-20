# Octree Cells Project

## Overview
The Octree Cells Project implements an octree structure to efficiently manage and query a three-dimensional array of cells. Each cell is represented by the `Cell` class, which contains properties such as typeID, index, normal, film, and potential. The octree structure allows for spatial partitioning of these cells, enabling efficient access and manipulation.

## Project Structure
```
octree-cells-project
├── src
│   ├── Cell.h          # Defines the Cell class representing a single cell in 3D space.
│   ├── OctreeNode.h    # Declares the OctreeNode class for the octree structure.
│   ├── OctreeNode.cpp  # Implements the methods for the OctreeNode class.
│   ├── main.cpp        # Entry point for the application, initializes Cells and constructs the octree.
│   └── types
│       └── index.h     # Defines additional types or structures used in the project.
├── CMakeLists.txt      # CMake configuration file for building the project.
└── README.md           # Documentation for the project.
```

## Building the Project
To build the project, follow these steps:

1. Ensure you have CMake installed on your system.
2. Open a terminal and navigate to the project directory.
3. Create a build directory:
   ```
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```
   cmake ..
   ```
5. Build the project:
   ```
   make
   ```

## Running the Application
After building the project, you can run the application by executing the generated binary in the build directory.

## Octree Structure
The octree is constructed by grouping cells in dimensions (1:4, 1:4, 1:4) as children and linking them to a parent node. This hierarchical structure allows for efficient spatial queries and management of the cells.

## Usage
The application initializes the `Cells` array, constructs the octree from the `Cells` data, and provides functionality for querying or visualizing the octree structure. Further details on usage will be provided in the documentation of the main application file.

## Contributing
Contributions to the project are welcome. Please feel free to submit issues or pull requests for any enhancements or bug fixes.