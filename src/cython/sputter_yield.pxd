# sputter_yield.pxd

cdef double sputter_yield_energy(double E, double Eth)

cdef public double[:,:] sputter_yield_angle(double gamma0, double gammaMax, double thetaMax)

cdef public double sputter_yield(
    double p0, double theta, double energy, double Eth)