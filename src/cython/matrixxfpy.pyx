cdef extern from "cpp_matrixxfpy.h":
    cdef cppclass MatrixXfPy:
        MatrixXfPy()
        MatrixXfPy(int d1, int d2)
        MatrixXfPy(MatrixXfPy other)
        int rows()
        int cols()
        float coeff(int, int)