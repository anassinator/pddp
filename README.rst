Probabilistic Differential Dynamic Programming
==============================================

.. image:: https://travis-ci.org/anassinator/pddp.svg?branch=master
  :target: https://travis-ci.org/anassinator/pddp

Probabilistic Differential Dynamic Programming (PDDP) is a data-driven,
probabilistic trajectory optimization framework for systems with unknown
dynamics. This is an implementation of Yunpeng Pan and Evangelos A. Theodorou's
paper in `PyTorch <https://pytorch.org>`_,
[`1 <https://papers.nips.cc/paper/5248-probabilistic-differential-dynamic-programming>`_].

Install
-------

To install simply clone and run:

.. code-block:: bash

  pip install .

You may also install the dependencies with `pipenv` as follows:

.. code-block:: bash

  pipenv install

Finally, you may add this to your own application with either:

.. code-block:: bash

  pip install 'git+https://github.com/anassinator/pddp.git#egg=pddp'
  pipenv install 'git+https://github.com/anassinator/pddp.git#egg=pddp'

Usage
-----

After installing, :code:`import` as follows:

.. code-block:: python

  import pddp

You can see the `notebooks <notebooks/>`_ directory for
`Jupyter <https://jupyter.org>`_ notebooks to see how common control problems
can be solved through PDDP.

Contributing
------------

Contributions are welcome. Simply open an issue or pull request on the matter.

Testing and Benchmarking
------------------------

You can run all unit tests and benchmarks through `pytest <https://pytest.org>`_
as follows:

.. code-block:: bash

  pytest

To speed things up, you may also run tests in parallel and disable benchmarks
with:

.. code-block:: bash

  pytest -n auto --benchmark-disable

You can install :code:`pytest` with:

.. code-block:: bash

  pipenv install --dev

Linting
-------

We use `YAPF <https://github.com/google/yapf>`_ for all Python formatting needs.
You can auto-format your changes with the following command:

.. code-block:: bash

  yapf --recursive --in-place --parallel .

You can install the formatter with:

.. code-block:: bash

  pipenv install --dev

License
-------

See `LICENSE <LICENSE>`_.
