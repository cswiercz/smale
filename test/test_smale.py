import pytest

from numpy import poly1d, abs
from smale import smale_newton


def test_newton():
    f = poly1d([1, 0, 0, -1])
    df = [f.deriv(k) for k in range(1, f.order+1)]
    r = f.r
    eps = 1e-6j

    xi = r[0]
    x = smale_newton(f, xi, df=df)
    assert abs(xi - x) < 1e-8

    xi = r[1]
    x = smale_newton(f, xi+eps, df=df)
    assert abs(xi - x) < 1e-8

    xi = r[2]
    x = smale_newton(f, xi+eps, df=df)
    assert abs(xi - x) < 1e-8


def test_warning_not_enough_derivatives():
    # test warning when not enough derivatives are provided
    f = poly1d([1, 0, 0, -1])

    with pytest.warns(RuntimeWarning):
        _ = smale_newton(f, 10.0)


def test_warning_alpha_condition():
    # test warning due to poor newton guess
    f = poly1d([1, 0, 0, -1])
    df = [f.deriv(k) for k in range(1, f.order+1)]

    with pytest.warns(RuntimeWarning):
        _ = smale_newton(f, 10.0, df=df)


def test_function_arguments():
    # test warning due to poor newton guess
    f = lambda x,a: x**3 - a
    df = [
        lambda x,a: 3*x**2,
        lambda x,a: 6*x,
        lambda x,a: 6
    ]
    x = smale_newton(f, 1.1, df=df, args=(1,))
    xi = 1.
    assert abs(xi - x) < 1e-8
