# include "cpp_matrixxfpy.h"
import numpy

cdef extern from "cpp_test.h": 
    cdef cppclass Test:
        Test()
        Test(int test1)
        int test1
        int returnFive()
        Test add "operator+"(Test other) 
        Test sub "operator-"(Test other)
        # MatrixXfPy getMatrixXf(int d1, int d2)

cdef extern from "cpp_matrixxfpy.h":
    cdef cppclass MatrixXfPy:
        MatrixXfPy()
        MatrixXfPy(int d1, int d2)
        MatrixXfPy(MatrixXfPy other)
        int rows()
        int cols()
        float coeff(int, int)
        MatrixXfPy getMatrixXf(int d1, int d2)

    # cdef cppclass Test:
    #     MatrixXfPy getMatrixXf(int d1, int d2)

cdef class pyTest: 
    cdef Test* thisptr # hold a C++ instance
    cdef cppclass Test:
        MatrixXfPy getMatrixXf(int d1, int d2)
    def __cinit__(self, int test1):
        self.thisptr = new Test(test1)
    def __dealloc__(self):
        del self.thisptr
 
    def __add__(pyTest left, pyTest other):
        cdef Test t = left.thisptr.add(other.thisptr[0])
        cdef pyTest tt = pyTest(t.test1)
        return tt
    def __sub__(pyTest left, pyTest other):
        cdef Test t = left.thisptr.sub(other.thisptr[0])
        cdef pyTest tt = pyTest(t.test1)
        return tt
 
    def __repr__(self): 
        return "pyTest[%s]" % (self.thisptr.test1)
 
    def returnFive(self):
        return self.thisptr.returnFive()
 
    def printMe(self):
        return "hello world"

    def getNDArray(self, int d1, int d2): 
        cdef MatrixXfPy me = self.thisptr.getMatrixXf(d1,d2) # get MatrixXfPy object
        
        result = numpy.zeros((me.rows(),me.cols())) # create nd array 
        # Fill out the nd array with MatrixXf elements 
        for row in range(me.rows()): 
            for col in range(me.cols()): 
                result[row, col] = me.coeff(row, col)   
        return result 




