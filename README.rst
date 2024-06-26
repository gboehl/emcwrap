emcwrap
=======

**Collection of tools for Bayesian inference using DIME MCMC sampling**

This provides a series of tools around the Differential-Independence Mixture Ensemble (DIME) MCMC sampler together with a nice set of statistical tools for Bayesian analysis. DIME MCMC is developed in `An Ensemble MCMC Sampler for Robust Bayesian Inference <https://gregorboehl.com/live/dime_mcmc_boehl.pdf>`_. *(Gregor Boehl, 2022, SSRN No. 4250395)*.

There exists a stand-alone implementations of the sampler in the ``dime_sampler`` `package <https://github.com/gboehl/dime_sampler>`_.

The sampler has a series of advantages over conventional samplers:

#. DIME MCMC is a (very fast) gradient-free **global multi-start optimizer** and, at the same time, a **MCMC sampler** that converges to the posterior distribution. This makes any posterior mode density maximization prior to MCMC sampling superfluous.
#. The DIME sampler is pretty robust for odd shaped, **multimodal distributions**.
#. DIME MCMC is **parallelizable**: many chains can run in parallel, and the necessary number of draws decreases almost one-to-one with the number of chains.
#. DIME proposals are generated from an **endogenous and adaptive proposal distribution**, thereby providing close-to-optimal proposal distributions for black box target distributions without the need for manual fine-tuning.

There is a nice `set of slides <https://gregorboehl.com/revealjs/emc>`_ on my website which explains the DIME principle.

.. figure:: https://github.com/gboehl/emcwrap/blob/main/docs/dist.png?raw=true
  :width: 800
  :alt: Sample and target distribution

  Figure: A trimodal example distribution in 35 dimensions

Installation
------------

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as typing

.. code-block:: bash

   pip install emcwrap

in your terminal or Anaconda Prompt.


Usage
-----

The proposal can be used directly as a drop-in replacement for `emcee <https://github.com/dfm/emcee>`_:

.. code-block:: python

    import emcee
    from dime_sampler import DIMEMove

    move = DIMEMove()

    ...
    def log_prob(x):
      ...
    # define your density function, the number of chains `nchain` etc...
    ...

    sampler = emcee.EnsembleSampler(nchain, ndim, log_prob, moves=move)
    ...
    # off you go sampling

The rest of the usage is hence analoge to Emcee, see e.g. `this tutorial <https://emcee.readthedocs.io/en/stable/tutorials/quickstart/>`_. The parameters specific to the ``DIMEMove`` are documented `here <https://emcwrap.readthedocs.io/en/latest/modules.html#module-emcwrap.moves>`_.

The provided tools for Bayesian analysis are ready-to-use, but largely undocumented. Find the module documentation here: https://emcwrap.readthedocs.io/en/latest/modules.html.


Tutorial
--------

Lets look at an example. Let's define a nice and challenging distribution (it's the distribution from the figure above):

.. code-block:: python

    # some import
    import emcwrap as ew
    import numpy as np
    import scipy.stats as ss
    from emcwrap.test_all import _create_test_func, _marginal_pdf_test_func

    # make it reproducible
    np.random.seed(0)

    # define distribution
    m = 2
    cov_scale = 0.05
    weight = (0.33, .1)
    ndim = 35
    initvar = np.sqrt(2)

    log_prob = _create_test_func(ndim, weight, m, cov_scale)

``log_prob`` will now return the log-PDF of a 35-dimensional Gaussian mixture with **three separate modes**.

Next, define the initial ensemble. In a Bayesian setup, a good initial ensemble would be a sample from the prior distribution. Here, we will go for a sample from a rather flat Gaussian distribution.

.. code-block:: python

    # number of chains and number of iterations
    nchain = ndim * 5
    niter = 5000

    # initial ensemble
    initmean = np.zeros(ndim)
    initcov = np.eye(ndim) * np.sqrt(2)
    initchain = ss.multivariate_normal(mean=initmean, cov=initcov).rvs(nchain)

Setting the number of parallel chains to ``5*ndim`` is a sane default. For highly irregular distributions with several modes you should use more chains. Very simple distributions can go with less.

Now let the sampler run for 5000 iterations.

.. code-block:: python

    # use the DIME proposal
    moves = ew.DIMEMove(aimh_prob=0.1, df_proposal_dist=10)
    sampler = ew.run_mcmc(log_prob, niter, p0=initchain, moves=moves)

.. code-block::

    [ll/MAF: 11.598(4e+00)/23%]: 100%|████████████████████ 5000/5000 [00:18<00:00, 164.70sample(s)/s]

The setting of ``aimh_prob`` is the actual default value. For less complex distributions (e.g. distributions closer to Gaussian) a higher value can be chosen, which accelerates burn-in. The information in the progress bar has the structure ``[ll/MAF: <maximum log-prob>(<standard deviation of log-prob>)/<mean acceptance fraction>]...``.

Note that if you wish to use emcee directly instead of the wrapper, you could simply do the following, which will give you the same result:

.. code-block:: python

    import emcee
    sampler = emcee.EnsembleSampler(nchain, ndim, log_prob, moves=moves)
    sampler.run_mcmc(initchain, int(niter), progress=True)


The following code creates the figure above, which is a plot of the marginal distribution along the first dimension (remember that this actually is a 35-dimensional distribution). For plotting, this tutorial assumes ``grgrlib`` to be installed (just run ``pip install grgrlib`` in terminal/prompt).

.. code-block:: python

    from grgrlib import figurator

    # get elements
    chain = sampler.get_chain()
    lprob = sampler.get_log_prob()

    # plotting
    figs, axs = figurator(1, 1, 1, figsize=(9,6))
    axs[0].hist(chain[-niter//2 :, :, 0].flatten(), bins=50, density=True, alpha=0.2, label="Sample")
    xlim = axs[0].get_xlim()
    x = np.linspace(xlim[0], xlim[1], 100)
    axs[0].plot(x, ss.norm(scale=np.sqrt(initvar)).pdf(x), "--", label="Initialization")
    axs[0].plot(x, ss.t(df=10, loc=moves.prop_mean[0], scale=moves.prop_cov[0, 0] ** 0.5).pdf(x), ":", label="Final proposals")
    axs[0].plot(x, _marginal_pdf_test_func(x, cov_scale, m, weight), label="Target")
    axs[0].legend()

To ensure proper mixing, let us also have a look at the MCMC traces, again focussing on the first dimension.

.. code-block:: python

    figs, axs = figurator(1, 1, 1)
    axs[0].plot(chain[:, :, 0], alpha=0.05, c="C0")

.. image:: https://github.com/gboehl/emcwrap/blob/main/docs/traces.png?raw=true
  :width: 800
  :alt: MCMC traces

Note how chains are also switching between the three modes because of the global proposal kernel.

While DIME is an MCMC sampler, it can straightforwardly be used as a global optimization routine. To this end, specify some broad starting region (in a non-Bayesian setup there is no prior) and let the sampler run for an extended number of iterations. Finally, assess whether the maximum value per ensemble did not change much in the last few hundred iterations. In a normal Bayesian setup, plotting the associated log-likelihood over time also helps to assess convergence to the posterior distribution.

.. code-block:: python

    figs, axs = figurator(1, 1, 1)
    axs[0].plot(lprob, alpha=0.05, c="C0")
    axs[0].plot(np.arange(niter), np.max(lprob) * np.ones(niter), "--", c="C1")

.. image:: https://github.com/gboehl/emcwrap/blob/main/docs/lprobs.png?raw=true
  :width: 800
  :alt: Log-likelihoods

References
----------

If you are using this software in your research, please cite

.. code-block:: bibtex

    @techreport{boehl2022mcmc,
    author={Gregor Boehl},
    title={Ensemble MCMC Sampling for Robust Bayesian Inference},
    journal={Available at SSRN 4250395},
    year={2022}
    }
