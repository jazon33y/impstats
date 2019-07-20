impstats
========

impstats is a Python package for the creation, manipulation,
and study of interval data and imprecise probability distributions.

- **Website (including documentation):** None
- **Mailing list:** None
- **Source:** None
- **Bug reports:** None

Using
-------
Tested with:

numpy==1.16.2
scipy==1.2.1
<!-- Install
-------

Install the latest version of impstats::

    $ pip install impstats

Install with all optional dependencies::

    $ pip install impstats[all] -->
Simple example
--------------

```python
>>> from impstats.dists import normal
>>> from impstats.interval import *
>>>
>>> normal_pbox = normal(5, 1)
>>> print(normal_pbox)
Pbox: ~ normal(range=[(1.9098, 8.0902), mean=5.0, var=1.0)
>>>
>>> big_int = I(1, 50)
>>> print(big_int)
Interval(1, 50)
>>> normal_pbox + big_int
Pbox: ~ (range=[(2.9098, 58.0902), mean=[6.0, 55.0], var=[0.0, 641.1335])
```

Bugs
----

Please report any bugs that you find [here](https://github.com/jazon33y/impstats/issues).
Or, even better, fork the repository on [GitHub](https://github.com/jazon33y/impstats)
and create a pull request (PR). We welcome all changes, big or small, and we
will help you make the PR if you are new to `git`.

