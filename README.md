# Smale

A collection of implementations of Smale's alpha theory for numerical root
finding.

    >>> from smale import smale_newton
    >>> f = lambda x: x**3 - 1
    >>> df = [lambda x: 3*x**2, lambda x: 6*x, lambda x: 6]
    >>> smale_newton(f, 1.1, df)
    1.0

## Dependencies

*Python Version*

* numpy
* scipy
* nosetests

*C++ Version* (coming soon)

## Tests

Use [nose](https://nose.readthedocs.org/en/latest) at the top level directory to
run tests:

    $ cd path/to/smale
    $ nosetests

## ToDo

* Implement C++11 version
* Analytic continuation: given function of two (or more?) variables with one (or
  n-1?) treated as a parameter, numerically determine the variable as a function
  of the parameter(s).
* Possible applications to dynamical systems, ODE solvers?
