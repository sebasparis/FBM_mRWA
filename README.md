# FBM_mRWA

This public resository contains two libraries that help solve open and closed quantum system dynamics.

The library floquet.h lets calculate several steady state parameters and explicit time evolution for an arbitrary quantum system with driven Hamiltonian of the form
$H(t) = V + A\cdot f(t)$, with $f(t)$ a funcition periodic in time.

The library FBM_mRWA also lets calculate steady state parameters and explicit time evolution but for the case of an open driven quantum system, weakly coupled to one
or several thermal baths so that the Born an Markov approximations can be appilied
