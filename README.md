impstats
========

impstats is a Python package for the creation, manipulation,
and study of interval data and imprecise probability distributions.

- **Website (including documentation):** http://no.there.is.none.com
- **Mailing list:** http://no.there.is.none.com
- **Source:** https://github.com/impstats/impstats
- **Bug reports:** https://github.com/impstats/impstats/issues

Install
-------

Install the latest version of impstats::

    $ pip install impstats

Install with all optional dependencies::

    $ pip install impstats[all]


Simple example
--------------

```python
>>> from impstats.dists import normal
>>> from impstats.interval import *
>>> normal_pbox = normal(5, 1)
>>> big_int = I(1, 50)
>>> 
['A', 'B', 'D']
```

Bugs
----

Please report any bugs that you find [here](https://github.com/impstats/impstats/issues).
Or, even better, fork the repository on [GitHub](https://github.com/impstats/impstats)
and create a pull request (PR). We welcome all changes, big or small, and we
will help you make the PR if you are new to `git`.

