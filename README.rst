emcwrap
=======

**Collection of tools for Bayesian inference using Adaptive Differential Evolution MCMC** 

This provides a nice set of statistical tools for Bayesian analisis, but at its core lies the ADEMC proposal developed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_. *(Gregor Boehl, 2022, CRC 224 discussion paper series)*.

Installation
------------

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as:

.. code-block:: bash

   pip install emcwrap
  
There exists a complementary stand-alone implementation in Julia language `here <https://github.com/gboehl/ADEMC.jl>`_.

   
Usage
-----

The proposal can be used directly as a drop-in replacement for `emcee <https://github.com/dfm/emcee>`_:

.. code-block:: python

    import emcee
    from emcwrap import ADEMove
    
    move = ADEMove(aimh_prob=.1, df_proposal_dist=10)
    
    ...
    # define your density function, number of chains etc...
    ...
    
    sampler = emcee.EnsembleSampler(nchain, ndim, log_prob, moves=move)
    ...
    # off you go sampling
 

The provided tools for Bayesian analysis are ready-to-use, but largely undocumented. Find the module documentation here: https://emcwrap.readthedocs.io/en/latest/modules.html

References
----------

.. code-block::

    @techreport{boehl2022mcmc,
    title         = {Ensemble MCMC Sampling for DSGE Models},
    author        = {Boehl, Gregor},
    year          = 2022,
    institution   = {CRC224 discussion paper series}
    }
