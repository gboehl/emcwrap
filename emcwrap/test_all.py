#!/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from .moves import DIMEMove
from .sampler import run_mcmc
from scipy.stats import multivariate_normal, norm

filepath = os.path.dirname(__file__)

def create_test_func(ndim, weight, distance, cov_scale):

    cov = np.eye(ndim)*cov_scale
    mean = np.zeros(ndim)
    mean[0] = distance/2

    lw0 = np.log(weight)
    lw1 = np.log(1-weight)

    dist = multivariate_normal(np.zeros(ndim), cov)

    def log_prob(p):
        return np.logaddexp(lw0 + dist.logpdf(p + mean), lw1 + dist.logpdf(p - mean))

    return log_prob


def marginal_pdf_test_func(x, cov_scale, m, weight):
    normal = norm(scale=np.sqrt(cov_scale))
    return (1-weight)*normal.pdf(x-m/2) + weight*normal.pdf(x+m/2)

def test_all(create=False):

    np.random.seed(0)

    # define distribution
    m = 2
    cov_scale = 0.05
    weight = 0.33
    ndim = 35

    log_prob = create_test_func(ndim, weight, m, cov_scale)

    moves = DIMEMove(aimh_prob=.1, df_proposal_dist=10)
    ndim = 35
    nchain = ndim*5
    niter = 300

    initmean = np.zeros(ndim)
    initcov = np.eye(ndim)*np.sqrt(2)
    initchain = multivariate_normal(mean=initmean, cov=initcov).rvs(nchain)

    sampler = run_mcmc(log_prob, niter, p0=initchain, moves=moves)
    chain = sampler.get_chain()

    path = os.path.join(filepath, "test_storage", "median.npy")
    median = np.median(chain[-int(niter/3):,:,0])

    if create:
        np.save(path, median)
        print(f'Test file updated at {path}')
    else:
        test_median = np.load(path)
        np.testing.assert_almost_equal(median, test_median, decimal=5)
