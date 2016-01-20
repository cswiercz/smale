r"""Smale's Alpha Theory :mod:`smale.smale`
=======================================

A Python implementation of Smale's alpha theory.

This module implements a version of :func:`scipy.optimize.newton` which will
warn the user if their guess `x0` to a root of a function `f` is not guarenteed
to be an approximate solution to `f`. (i.e. not guarenteed to lie in the
quadratic convergence basin of some root of f.)

Functions
---------

.. autosummary::

    smale_newton
    smale_alpha
    smale_beta
    smale_gamma

References
----------

.. [Smale] S. Smale, "Newton's method estimates from data at one point",
   Springer New York, 1986.

.. [AlphaCertified] J. D. Hauenstein, F. Sottile, "AlphaCertified: certifying
   solutions to polynomial systems", ACM Trans. Math. Softw., vol. 38, no. 4,
   pp. 1-20, 2012.

Examples
--------

Contents
--------

"""

import numpy
import numpy.lib.polynomial
import warnings

from scipy.optimize import newton

def smale_newton(f, x0, df=(), args=(), tol=1.48e-8, maxiter=50):
    r"""
    Find a zero using Smale's alpha theory.

    Find a zero of the function `func` given a nearby starting point `x0` using
    Newton's method. Smale's alpha theory is used to certify that `x0` is an
    approximate solution as long as at least two derivatives of `f` are
    provided in `df`.

    If none or only one derivative is given in `df` then a warning is raised
    and Newton's method will run without any certification.

    Parameters
    ----------
    func : function
        The function whose zero is wanted. It must be a function of a single
        variable of the form f(x,a,b,c...), where a,b,c... are extra arguments
        that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    df : list of functions, optional
        A list of derivatives of the function beyond the first. The more that
        can be provided the more likely Smale's alpha theory will work in the
        sense that too few derivatives may underestimate the value of
        alpha(f,x).
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    zero : float
        Estimated location where function is zero.
    """
    # skip using smale alpha if not enough derivatives are provided
    if len(df) < 2:
        warnings.warn('not enough derivatives provided for smale',
                      RuntimeWarning)
    else:
        # raise a warning if x0 is not an approximate solution
        alpha = smale_alpha(f, x0, df, args=args)
        if alpha > 0.15767078078675478:
            warnings.warn('the estimate may not be an approximate solution to the '
                          'function', RuntimeWarning)

    # newton iterate (omit use of fprime2 due to unknown issues)
    fprime = df[0] if len(df) > 0 else None
    zero = newton(f, x0, fprime=fprime, args=args, tol=tol, maxiter=maxiter)
    return zero

def smale_alpha(f, x0, df, args=()):
    r"""
    Compute the Smale alpha function.

    Parameters
    ----------
    func : function
        The function whose zero is wanted. It must be a function of a single
        variable of the form f(x,a,b,c...), where a,b,c... are extra arguments
        that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    df : list of functions
        A list of additional order derivatives of the function beyond the
        first. The more that can be provided the more likely Smale's alpha
        theory will work in the sense that too few derivatives may
        underestimate the value of alpha(f,x).
    args : tuple, optional
        Extra arguments to be used in the function call.
    """
    alpha = smale_beta(f,x0,df,args=args) * smale_gamma(f,x0,df,args=args)
    return alpha

def smale_beta(f, x0, df, args=()):
    r"""
    Compute the Smale alpha function.

    Parameters
    ----------
    func : function
        The function whose zero is wanted. It must be a function of a single
        variable of the form f(x,a,b,c...), where a,b,c... are extra arguments
        that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    df : list of functions
        A list of additional order derivatives of the function beyond the
        first. The more that can be provided the more likely Smale's alpha
        theory will work in the sense that too few derivatives may
        underestimate the value of alpha(f,x).
    args : tuple, optional
        Extra arguments to be used in the function call.
    """
    _args = (x0,) + args
    beta = numpy.abs(f(*_args)/df[0](*_args))
    return beta

def smale_gamma(f, x0, df, args=()):
    r"""
    Compute the Smale alpha function.

    Parameters
    ----------
    func : function
        The function whose zero is wanted. It must be a function of a single
        variable of the form f(x,a,b,c...), where a,b,c... are extra arguments
        that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    df : list of functions
        A list of additional order derivatives of the function beyond the
        first. The more that can be provided the more likely Smale's alpha
        theory will work in the sense that too few derivatives may
        underestimate the value of alpha(f,x).
    args : tuple, optional
        Extra arguments to be used in the function call.
    """
    _args = (x0,) + args
    gammas = numpy.zeros(len(df)-1, dtype=numpy.double)

    scale = 1./df[0](*_args)
    for k in range(1,len(df)):
        scale /= 1.0*(k+1)  # efficiently compute 1/(dfx0 * k!)
        gammas[k-1] = numpy.abs(scale * df[k](*_args))**(1./k)

    gamma = gammas.max()
    return gamma
