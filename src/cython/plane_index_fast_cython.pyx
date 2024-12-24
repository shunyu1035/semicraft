import numpy as np
cimport numpy as cnp

def plane_index_fast_cython(cnp.ndarray[cnp.float64_t, ndim=4] film_label_index_normal):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] plane_data
    cdef cnp.ndarray[cnp.float64_t, ndim=2] vacuum_data
    cdef cnp.ndarray[cnp.float64_t, ndim=3] labels
    cdef cnp.ndarray[cnp.int64_t, ndim=2] plane_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=2] vacuum_indices
    cdef int i, j, k

    labels = film_label_index_normal[:, :, :, 0]

    cdef cnp.ndarray[cnp.uint8_t, ndim=3] plane_bool = labels == 1
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] vacuum_bool = labels == -1

    plane_indices = np.argwhere(plane_bool)
    vacuum_indices = np.argwhere(vacuum_bool)

    plane_data = np.empty((plane_indices.shape[0], film_label_index_normal.shape[3]), dtype=np.float64)
    vacuum_data = np.empty((vacuum_indices.shape[0], film_label_index_normal.shape[3]), dtype=np.float64)

    for i in range(plane_indices.shape[0]):
        plane_data[i, :] = film_label_index_normal[plane_indices[i, 0], plane_indices[i, 1], plane_indices[i, 2], :]

    for i in range(vacuum_indices.shape[0]):
        vacuum_data[i, :] = film_label_index_normal[vacuum_indices[i, 0], vacuum_indices[i, 1], vacuum_indices[i, 2], :]

    return plane_data, vacuum_data
