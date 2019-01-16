# Smale

A Python implementation of Smale's alpha theory for numerical root finding.

```python
>>> from smale import smale_newton
>>> f = lambda x: x**3 - 1
>>> df = [lambda x: 3*x**2, lambda x: 6*x, lambda x: 6]
>>> smale_newton(f, 1.1, df)
1.0
>>> smale_newton(f, -0.5 + 1.j, df)
(-0.5+0.8660254037844386j)
>>> smale_newton(f, -0.5 - 1.j, df)
(-0.5+0.8660254037844386j)
```

## Installation

Smale depends on the following Python packages:
* numpy
* scipy
* pytest

For a local install run:

```sh
$ cd /path/to/smale
$ python setup.py install --user
```

(Omit `--user` for a system-wide installation.)

## Testing

Use [pytest](https://pytest.readthedocs.org/en/latest) at the top-level directory to
run tests:

```sh
$ cd /path/to/smale
$ pytest
```
