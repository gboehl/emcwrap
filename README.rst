emcwrap
=======

**Collection of tools for Bayesian inference using DIME MCMC**

This provides the Differential-Independence Mixture Ensemble (DIME) proposal together with a nice set of statistical tools for Bayesian analysis. DIME MCMC (previously ADEMC) is developed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_. *(Gregor Boehl, 2022, CRC 224 discussion paper series)*.

The sampler has a series of advantages over conventional samplers:

#. At core, DIME MCMC is a (very fast) **global multi-start optimizer** that converges to the posterior distribution. This makes any posterior mode density maximization prior to MCMC sampling superfluous.
#. The DIME sampler is pretty robust for odd shaped, **bimodal distributions**.
#. DIME MCMC is **parallelizable**: many chains can run in parallel, and the necessary number of draws decreases almost one-to-one with the number of chains.
#. DIME proposals are generated from an **endogenous and adaptive proposal distribution**, thereby reducing the number of necessary meta-parameters and providing close-to-optimal proposal distributions.

Installation
------------

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as:

.. code-block:: bash

   pip install emcwrap
  
There exists a complementary stand-alone implementation in `Julia language <https://github.com/gboehl/ADEMC.jl>`_.

   
Usage
-----

The proposal can be used directly as a drop-in replacement for `emcee <https://github.com/dfm/emcee>`_:

.. code-block:: python

    import emcee
    from emcwrap import DIMEMove
    
    move = DIMEMove(aimh_prob=.1, df_proposal_dist=10)
    
    ...
    # define your density function, number of chains etc...
    ...
    
    sampler = emcee.EnsembleSampler(nchain, ndim, log_prob, moves=move)
    ...
    # off you go sampling
 
The rest of the usage is hence analoge to Emcee, see e.g. `this tutorial <https://emcee.readthedocs.io/en/stable/tutorials/quickstart/>`_. The parameters specific to the ``ADEMove`` are documented `here <https://emcwrap.readthedocs.io/en/latest/modules.html#module-emcwrap.moves>`_.

The provided tools for Bayesian analysis are ready-to-use, but largely undocumented. Find the module documentation here: https://emcwrap.readthedocs.io/en/latest/modules.html

References
----------

If you are using this software in your research, please cite

.. code-block::

    @techreport{boehl2022mcmc,
    title         = {Ensemble MCMC Sampling for DSGE Models},
    author        = {Boehl, Gregor},
    year          = 2022,
    institution   = {CRC224 discussion paper series}
    }
