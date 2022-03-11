#!/bin/python
# -*- coding: utf-8 -*-

import os
import tqdm
import emcee
import numpy as np
from grgrlib import map2arr
from .stats import summary


def run_mcmc(lprob, p0, nsteps, moves=None, priors=None, backend=None, update_freq=None, resume=False, pool=None, report=None, description=None, temp=1, maintenance_interval=False, seed=None, verbose=False, **kwargs):

    if seed is None:
        seed = 0

    np.random.seed(seed)

    if isinstance(backend, str):
        backend = emcee.backends.HDFBackend(
            os.path.splitext(backend)[0] + '.h5')

    if resume:
        p0 = backend.get_chain()[-1]

    nwalks, ndim = np.shape(p0)

    if update_freq is None:
        update_freq = nsteps // 5

    sampler = emcee.EnsembleSampler(
        nwalks, ndim, lprob, moves=moves, pool=pool, backend=backend)

    if not verbose:  # verbose means VERY verbose
        np.warnings.filterwarnings("ignore")

    if verbose > 2:
        report = report or print
    else:
        pbar = tqdm.tqdm(total=nsteps, unit="sample(s)", dynamic_ncols=True)
        report = report or pbar.write

    old_tau = np.inf
    cnt = 0

    for result in sampler.sample(p0, iterations=nsteps, **kwargs):

        if not verbose:
            lls = list(result)[1]
            maf = np.mean(sampler.acceptance_fraction) * 100
            pbar.set_description(
                "[ll/MAF:%s(%1.0e)/%1.0f%%]" % (str(np.max(lls))
                                                [:7], np.std(lls), maf)
            )

        if cnt and update_freq and not cnt % update_freq:

            prnttup = "(mcmc:) Summary from last %s of %s iterations" % (
                update_freq, cnt)

            if temp < 1:
                prnttup += " with temp of %s%%" % (np.round(temp * 100, 6))

            if description is not None:
                prnttup += " (%s)" % str(description)

            prnttup += ":"

            report(prnttup)

            sample = sampler.get_chain()
            lprobs = sampler.get_log_prob(flat=True)
            acfs = sampler.acceptance_fraction

            tau = emcee.autocorr.integrated_time(sample, tol=0)
            min_tau = np.min(tau).round(2)
            max_tau = np.max(tau).round(2)
            dev_tau = np.max(np.abs(old_tau - tau) / tau)

            tau_sign = ">" if max_tau > sampler.iteration / 50 else "<"
            dev_sign = ">" if dev_tau > 0.01 else "<"

            if priors is not None:
                mcmc_summary(
                    chain=sample[-update_freq:],
                    lprobs=lprobs[-update_freq:],
                    priors=priors,
                    acceptance_fraction=acfs[-update_freq:],
                    out=lambda x: report(str(x)),
                )

            report(
                "Convergence stats: tau is in (%s,%s) (%s%s) and change is %s (%s0.01)."
                % (
                    min_tau,
                    max_tau,
                    tau_sign,
                    sampler.iteration / 50,
                    dev_tau.round(3),
                    dev_sign,
                )
            )

        if cnt and update_freq and not (cnt + 1) % update_freq:
            sample = sampler.get_chain()
            old_tau = emcee.autocorr.integrated_time(sample, tol=0)

        if not verbose:
            pbar.update(1)

        # avoid mem leakage
        if maintenance_interval and cnt and pool and not cnt % maintenance_interval:
            pool.clear()

        cnt += 1

    pbar.close()
    if pool:
        pool.close()

    if not verbose:
        np.warnings.filterwarnings("default")

    if backend is None:
        return sampler
    else:
        return backend


def mcmc_summary(
    chain,
    lprobs,
    priors,
    acceptance_fraction=None,
    out=print,
    **args
):

    nchain = chain.reshape(-1, chain.shape[-1])
    lprobs = lprobs.reshape(-1, lprobs.shape[-1])
    mode_x = nchain[lprobs.argmax()]

    res = summary(priors, chain, mode_x, **args)

    out(res.round(3))

    if acceptance_fraction is not None:

        out("Mean acceptance fraction:" +
            str(np.mean(acceptance_fraction).round(3)).rjust(13))

    return res


def get_prior_sample(frozen_prior, nsamples, check_func=False, seed=None, mapper=map, verbose=True):

    if seed is None:
        seed = 0

    if check_func and not callable(check_func):
        raise Exception('`check_func` must be `False` or a callable')

    def runner(locseed):

        np.random.seed(seed + locseed)
        done = False
        no = 0

        while not done:

            no += 1

            with np.warnings.catch_warnings(record=False):
                try:
                    np.warnings.filterwarnings("error")
                    rst = np.random.randint(2 ** 31)  # win explodes with 2**32
                    pdraw = [
                        pl.rvs(random_state=rst + sn)
                        for sn, pl in enumerate(frozen_prior)
                    ]

                    if check_func:
                        draw_prob = check_func(pdraw)
                        done = not np.any(np.isinf(draw_prob))
                    else:
                        done = True

                except Exception as e:
                    if verbose > 1:
                        print(str(e) + " (%s) " % no)
                    if not locseed and no == 10:
                        raise

        return pdraw, no

    if verbose > 1:
        print("(prior_sample:) Sampling from the pior...")

    wrapper = tqdm.tqdm if verbose < 2 else (lambda x, **kwarg: x)
    pmap_sim = wrapper(mapper(runner, range(nsamples)), total=nsamples)

    draws, nos = map2arr(pmap_sim)

    if verbose and check_func:
        print("(prior_sample:) Sampling done. Check fails for %2.2f%% of the prior."
              % (100 * (sum(nos) - nsamples) / sum(nos))
              )

    return draws
