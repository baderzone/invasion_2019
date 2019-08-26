#!/usr/bin/env python
#
# Copyright (C) 2017 Joel S. Bader
# You may use, distribute, and modify this code
# under the terms of the Python Software Foundation License Version 2
# available at https://www.python.org/download/releases/2.7/license/
#
import argparse
import os
import fnmatch
import math
import string
import copy

import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.stats import rankdata
from scipy.stats import pearsonr
from scipy.stats import f as f_distribution
from scipy.stats import chi2 as chisq_distribution
from scipy.stats import norm as norm_distribution
from scipy.stats import ttest_ind as ttest_ind
from scipy.stats import ttest_rel as ttest_rel
from scipy.stats import ttest_1samp as ttest_1samp
from numpy import random as nprand
#import pandas as pd

import matplotlib
matplotlib.use("TKAgg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
import matplotlib.colors as mpcolors

import statsmodels.api as sm

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis2d')
logger.setLevel(logging.INFO)

MODELS = ['zero', 'one', 'two']

def read_table(filename, delim='\t'):
    logger.info('reading table from %s', filename)
    fp = open(filename, 'r')
    ret = dict()
    orig_order = [ ]
    header = fp.readline()
    cols = header.strip().split(delim)
    fields = cols[1:]
    ntok = len(cols)
    for line in fp:
        toks = line.strip().split(delim)
        if toks[0][0] == '#':
            logger.info('skipping comment line: %s' % line)
            continue
        if (len(toks) != ntok):
            logger.warn('bad token count, %d instead of %d: %s', len(toks), ntok, line)
        nmissing = ntok - len(toks)
        if nmissing > 0:
            toks = toks + ([''] * nmissing)
        k = toks[0]
        assert(k not in ret), 'repeated row: %s' % k
        ret[k] = dict()
        for (k1, v1) in zip( fields, toks[1:]):
            ret[k][k1] = v1
        orig_order.append(k)
    logger.info('%s: %d rows, %d columns', filename, len(ret), ntok)
    return(ret)
    #return(ret, orig_order)


def test_scatter_pool(y_vec, x_vec, group_vec, yname, xname, tailfrac_line, tailfrac_color, fullpath, mingroupsize=5, plot='scatter'):

    # logger.info('testing %s ~ %s, tailfrac %f, mingroupsize %d, transform %s', yname, xname, tailfrac, mingroupsize, transform)
    assert(mingroupsize > 1), 'need at least 2 organoids for tails but mingroupsize = %d' % mingroupsize

    # group rows according to column byname
    group_cnt = dict()
    for g in group_vec:
        group_cnt[g] = group_cnt.get(g, 0) + 1
    groups = sorted(group_cnt.keys())
    # logger.info('found %d groups', len(groups))
    largegroups = [ g for g in groups if group_cnt[g] >= mingroupsize ]
    # logger.info('found %d large groups', len(largegroups))
    groups = sorted(largegroups)

    # subset the data to large groups
    x_subset = [ ]
    y_subset = [ ]
    g_subset = [ ]
    for (x, y, g) in zip(x_vec, y_vec, group_vec):
        if g not in groups:
            continue
        x_subset.append(x)
        y_subset.append(y)
        g_subset.append(g)

    logger.info('*** tests use %d tumors with %d organoids ***', len(groups), len(x_subset))

    # rename
    x_vec = x_subset
    y_vec = y_subset
    group_vec = g_subset

    x_sum = dict()
    y_sum = dict()
    for (g, y, x) in zip(group_vec, y_vec, x_vec):
        x_sum[g] = x_sum.get(g, 0) + x
        y_sum[g] = y_sum.get(g, 0) + y
    
    x_avg = dict()
    y_avg = dict()
    for g in groups:
        n = float(group_cnt[g])
        x_avg[g] = x_sum[g] / n
        y_avg[g] = y_sum[g] / n


    xx_between = [x_avg[g] for g in groups]
    yy_between = [y_avg[g] for g in groups]

    xx_within = [ ]
    yy_within = [ ]
    for (g, y, x) in zip(group_vec, y_vec, x_vec):
        if g not in groups:
            continue
        dx = x - x_avg[g]
        dy = y - y_avg[g]
        xx_within.append(dx)
        yy_within.append(dy)

    logger.info('sum of within: y %f\tx %f', sum(yy_within), sum(xx_within))

    # first have to collect the (dx, dy) values for each group
    group_to_yx = dict()
    for (g, y, x) in zip(group_vec, y_vec, x_vec):
        if g not in groups:
            continue
        dx = x - x_avg[g]
        dy = y - y_avg[g]
        group_to_yx[g] = group_to_yx.get(g, [ ]) + [ (dy, dx) ]
        # group_to_yx[g] = group_to_yx.get(g, [ ]) + [ (y, x) ] # the tail pools won't have individual means to subtract

    frac_list = [ ]
    tailpval_list = [ ]
    poolpval_list = [ ]

    dx_list = [ ]
    dy_list = [ ]
    color_list = [ ]
    for g in groups:
        n = len(group_to_yx[g])
        yx_list = sorted(group_to_yx[g])
        x_group = [ x for (y, x) in yx_list ]
        y_group = [ y for (y, x) in yx_list ]
        col_group = ['k'] * n
        for (frac, col) in tailfrac_color:
            nlower = math.ceil(frac * n)
            nupper = nlower
            if (nupper + nlower > n):
                nupper = nupper - 1
            for i in range(nlower):
                col_group[i] = col
            for i in range(n-nupper, n):
                col_group[i] = col
        dx_list = dx_list + x_group
        dy_list = dy_list + y_group
        color_list = color_list + col_group
            
    for tailfrac in tailfrac_line:
        xx_lower = [ ]
        yy_lower = [ ]
        xx_upper = [ ]
        yy_upper = [ ]
        group_lowermean = [ ]
        group_uppermean = [ ]
        group_diff = [ ]
        for g in groups:
            n = len(group_to_yx[g])
            nlower = math.ceil(tailfrac * n)
            nupper = nlower
            if (nlower + nupper > n):
                nupper = nupper - 1
                # nlower = nlower - 1
                # logger.info('frac %f n %d nl %d nu %d', tailfrac, n, nlower, nupper)
            assert(nlower + nupper <= n), 'tails too large n %d nl %d nu %d' % (n, nlower, nupper)
            assert(nlower > 0), 'empty lower tail'
            assert(nupper > 0), 'empty upper tail'
            # logger.info('group %s has %d organoids, keeping most extreme %d', g, n, ntail)
            yx_list = sorted(group_to_yx[g])
            sumlower = 0.0
            for (dy, dx) in yx_list[:nlower]:
                xx_lower.append(dx)
                yy_lower.append(dy)
                sumlower += dx
            sumhigher = 0.0
            for (dy, dx) in yx_list[n-nupper:]:
                xx_upper.append(dx)
                yy_upper.append(dy)
                sumhigher += dx
            lowermean = sumlower / float(nlower)
            uppermean = sumhigher / float(nupper)
            group_lowermean.append(lowermean)
            group_uppermean.append(uppermean)
            group_diff.append(uppermean - lowermean)
            
        xx_tail = xx_lower + xx_upper
        yy_tail = yy_lower + yy_upper

        (tail_t, tail_tpval) = ttest_ind(xx_upper, xx_lower)
        (pool_tind, pool_tpvalind) = ttest_ind(group_uppermean, group_lowermean)
        (pool_trel, pool_tpvalrel) = ttest_rel(group_uppermean, group_lowermean)
        (diff_t, diff_tpval) = ttest_1samp(group_diff, popmean=0)
        # logger.info('*** %s ttest frac %f\ttail %.2g\tpool %.2g\tdiff %.2g ***', xname, tailfrac, tail_tpval, pool_tpval, diff_tpval)
        
        fittail = sm.OLS(yy_tail,sm.add_constant(xx_tail),hasconst=True).fit() # be careful because there could be a constant term

        logger.info('*** %s ttest frac %f\ttail %g\tpool_ind %g\tpool_rel %g ***', xname, tailfrac, fittail.pvalues[1], pool_tpvalind, pool_tpvalrel)
            
        group_neg = [-1.0] * len(group_lowermean)
        group_pos = [1.0] * len(group_uppermean)
        xx_pooled = group_lowermean + group_uppermean
        yy_pooled = group_neg + group_pos

        # fitpool = sm.OLS(yy_pooled,sm.add_constant(xx_pooled),hasconst=True).fit() 
        frac_list.append(tailfrac)
        tailpval_list.append(fittail.pvalues[1])
        poolpval_list.append(pool_tpvalrel)


    ### plotting ###
    
    plt.figure(figsize=(18,5))
        
    (xlabel, ylabel) = (xname, yname)
    myfs = 'x-large'
    myls = 'large'

    ### panel B: individual organoids in tails ###

    fitw = sm.OLS(yy_within,xx_within,hasconst=False).fit()
    pvalw = fitw.pvalues[0]
    deltaw = max(xx_within) - min(xx_within)
    xx_fitw = np.linspace(min(xx_within) - 0.1 * deltaw, max(xx_within) + 0.1*deltaw,100)
    yy_fitw = xx_fitw*fitw.params[0]

    logger.info('\n*** within %s %s:***\n%s', ylabel, xlabel, fitw.summary())

    ax = plt.subplot(132)
    ax.tick_params(labelsize=myls)
    plt.title('(B) Within-tumor test, $R^2$ = %.2f, $p$ = %.2g' % (fitw.rsquared, pvalw), fontsize=myfs )
    plt.xlabel('$\Delta$' + ' ' + xlabel, fontsize=myfs)
    plt.ylabel('$\Delta$' + ' ' + ylabel, fontsize=myfs)

    logger.info('tailfrac_color %s', str(list(tailfrac_color)))
    for (frac, col) in sorted(tailfrac_color):
        myx = [ ]
        myy = [ ]
        myc = [ ]
        for (x,y,c) in zip(dx_list, dy_list, color_list):
            if (c == col):
                myx.append(x)
                myy.append(y)
                myc.append(col)
        plt.scatter(myx, myy, color=myc, marker='.', label='%d%% tails' % round(100 * frac))
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.plot(xx_fitw, yy_fitw, 'k--')

    ### Panel C: extreme tails and pools ###
    
    ax = plt.subplot(133)
    ax.tick_params(labelsize=myls)
    plt.title('(C) Within-tumor test, extreme tails and pooling', fontsize=myfs)
    plt.xlabel('Tail fraction, $\Delta$ ' + yname, fontsize=myfs)
    plt.ylabel('$-\log_{10}\ P$-value, $\Delta$ ' + xname, fontsize=myfs)
    
    plt.plot(frac_list, -np.log10(tailpval_list), 'k-', label='Tail organoids')
    plt.plot(frac_list, -np.log10(poolpval_list), 'k--', label='Tail pools')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)

    ### Panel A: between-tumor test ###
        
    fitb = sm.OLS(yy_between,sm.add_constant(xx_between),hasconst=True).fit()
    pvalb = fitb.pvalues[1]
    deltab = max(xx_between) - min(xx_between)
    xx_fitb = np.linspace(min(xx_between) - 0.1 * deltab, max(xx_between) + 0.1*deltab,100)
    yy_fitb = xx_fitb*fitb.params[1] + fitb.params[0]

    logger.info('\n*** between %s %s:***\n%s', ylabel, xlabel, fitb.summary())

    ax = plt.subplot(131)
    ax.tick_params(labelsize=myls)
    plt.title('(A) Between-tumor test, $R^2$ = %.2f, $p$ = %.2g' % (fitb.rsquared, pvalb), fontsize=myfs )
    plt.xlabel(xlabel, fontsize=myfs)
    plt.ylabel(ylabel, fontsize=myfs)

    plt.scatter(xx_between, yy_between, marker='o', color='k')
    plt.plot(xx_fitb, yy_fitb, 'k--')
    if xname == 'organoid area':
        plt.legend(loc='best', fancybox=True, framealpha=0.5, fontsize='x-small')

    plt.savefig(fullpath)
        
    return None

def get_loglik(group_vec, x_vec, name):
    g2n = dict()
    g2sum = dict()
    for (g, x) in zip(group_vec, x_vec):
        g2n[g] = g2n.get(g, 0) + 1.0
        g2sum[g] = g2sum.get(g, 0) + x
    groups = sorted(g2n.keys())
    g2mean = dict()
    for g in groups:
        g2mean[g] = g2sum[g] / g2n[g]
    g2sigmasq = dict()
    sigmasqtot = 0.0
    for (g, x) in zip(group_vec, x_vec):
        deltasq = (x - g2mean[g])**2
        sigmasqtot += deltasq
        g2sigmasq[g] = g2sigmasq.get(g, 0) + deltasq
        # logger.info('g %s sigmasq %g', g, sigmasqtot)
    for g in groups:
        g2sigmasq[g] = g2sigmasq[g] / g2n[g]
    loglik = 0.0
    for g in groups:
        sigmasq = g2sigmasq[g]
        if sigmasq <= 0:
            logger.info('skipping group %s because sigmasq non-positive %g', str(g), sigmasq)
            continue
        loglik = loglik - 0.5 * g2n[g] * math.log(2.0 * math.pi * sigmasq)
        loglik = loglik - 0.5 * g2n[g]
        loglik = loglik - math.log(g2n[g]) # BIC correction for mean and variance parameters
    llgroup = loglik
        
    # alternative loglik where all groups share sigma
    llshare = 0.0
    ntot = 0.0
    for g in groups:
        ntot += g2n[g]
    logger.info('sigmasqtot %g ntot %g', sigmasqtot, ntot)
    sigmasqshare = sigmasqtot / float(ntot)
    llshare = -0.5 * ntot * math.log(2.0 * math.pi * sigmasqshare) - 0.5 * ntot
    llshare = llshare - 0.5 * math.log(ntot) # BIC for shared variance
    logger.info('first part of llshare %g, sigmasqshare %g', llshare, sigmasqshare)
    for g in groups:
        llshare = llshare - 0.5 * math.log(g2n[g]) # BIC for mean of each group
    logger.info('final llshare %g', llshare)
    
    # calculate probability of shared model vs group model
    logratio = llshare - llgroup
    probratio = math.exp(logratio)
    probshare = probratio / (1.0 + probratio)
    probgroup = 1.0 / (1.0 + probratio)
    logger.info('scale %s probratio %g probshare %g probgroup %g', name, probratio, probshare, probgroup)
    
    logger.info('loglik shared %g individual %g', llshare, loglik)
    return(llshare, loglik)
    

def compare_scales(group_vec, a_vec, b_vec, aname, bname):
    logger.info('comparing model %s vs model %s', aname, bname)
    (lla1, lla2) = get_loglik(group_vec, a_vec, aname)
    (llb1, llb2) = get_loglik(group_vec, b_vec, bname)
    for (lla, llb, method) in [ (lla1, llb1, 'shared variance'), (lla2, llb2, 'group variance')]:
        logger.info('model selection using %s', method)
        logger.info('lla %g llb %g', lla, llb)
        logratio = llb - lla
        probratio = math.exp(logratio)
        logger.info('log(b/a) = %g pr(b/a) = %g', logratio, probratio)
        proba = 1.0/(1.0 + probratio)
        probb = probratio * proba
        logger.info('proba %g probb %g', proba, probb)
    return None

def get_group_stats(group_vec, x_vec):

    g2n = dict()
    g2sum = dict()
    for (g, x) in zip(group_vec, x_vec):
        g2n[g] = g2n.get(g, 0) + 1.0
        g2sum[g] = g2sum.get(g, 0) + x
    groups = sorted(g2n.keys())

    g2mean = dict()
    for g in groups:
        g2mean[g] = g2sum[g] / g2n[g]
    g2sigmasq = dict()
    for (g, x) in zip(group_vec, x_vec):
        deltasq = (x - g2mean[g])**2
        g2sigmasq[g] = g2sigmasq.get(g, 0) + deltasq
    for g in groups:
        g2sigmasq[g] = g2sigmasq[g] / g2n[g]

    return(groups, g2n, g2mean, g2sigmasq)

def write_group_stats(filename, groups, g2n, g2mean, g2sigmasq):
    fp = open(filename,'w')
    for g in groups:
        fp.write('\t'.join( [ g ] + [ str(data[g]) for data in [g2n, g2mean, g2sigmasq ] ]) + '\n')
    fp.close()
    
def get_scores(group_vec_orig, x_vec_orig, minsize=1):
    # model zero: treat as a single group
    # model one: each group has its own mean, but groups share a variance
    # model two: each group has its own mean and variance

    # logger.info('original len group %d len x %d', len(group_vec_orig), len(x_vec_orig))
    
    g2n = dict()
    for g in group_vec_orig:
        g2n[g] = g2n.get(g, 0) + 1.0
    group_vec = [ ]
    x_vec = [ ]
    for (g, x) in zip(group_vec_orig, x_vec_orig):
        if g2n[g] < minsize:
            continue
        group_vec.append(g)
        x_vec.append(x)
    seen = dict()
    for g in group_vec:
        seen[g] = True

    logger.info('minsize %d gives %d groups, %d observations', minsize, len(seen), len(x_vec))
    # logger.info('len group %d len x %d', len(group_vec), len(x_vec))
    score_dict = dict()
    for m in MODELS:
        score_dict[m] = 0.0
    tot_n = float(len(x_vec))
    logger.info('tot_n %g', tot_n)
    tot_sum = sum(x_vec)
    tot_mean = tot_sum / tot_n
    delsq_vec = [ (x - tot_mean)**2 for x in x_vec ]
    tot_sigmasqsum = sum(delsq_vec)
    tot_sigmasq = tot_sigmasqsum / tot_n
    score_dict['zero'] = - 0.5 * tot_n * math.log(2.0 * math.pi * tot_sigmasq) - (0.5 * tot_n) - math.log(tot_n)

    g2n = dict()
    g2sum = dict()
    for (g, x) in zip(group_vec, x_vec):
        g2n[g] = g2n.get(g, 0) + 1.0
        g2sum[g] = g2sum.get(g, 0) + x
    groups = sorted(g2n.keys())

    g2mean = dict()
    for g in groups:
        g2mean[g] = g2sum[g] / g2n[g]
    g2sigmasq = dict()
    share_sigmasq = 0.0
    sharesq_vec = [ ]
    for (g, x) in zip(group_vec, x_vec):
        deltasq = (x - g2mean[g])**2
        share_sigmasq += deltasq
        sharesq_vec.append(deltasq)
        g2sigmasq[g] = g2sigmasq.get(g, 0) + deltasq
    share_sigmasq = share_sigmasq / tot_n
    for g in groups:
        g2sigmasq[g] = g2sigmasq[g] / g2n[g]

    logger.info('tot_sigmasq %g share_sigmasq %g', tot_sigmasq, share_sigmasq)        
    score_dict['one'] = -0.5 * tot_n * math.log(2.0 * math.pi * share_sigmasq) - (0.5 * tot_n) - (0.5 * math.log(tot_n))
    for g in groups:
        score_dict['one'] += -0.5 * math.log(g2n[g])

    for g in groups:
        sigmasq = g2sigmasq[g]
        if sigmasq <= 0:
            logger.info('using shared sigmasq for group %s because sigmasq non-positive %g n %f', str(g), sigmasq, g2n[g])
            sigmasq = share_sigmasq
        score_dict['two'] += - 0.5 * g2n[g] * math.log(2.0 * math.pi * sigmasq) - (0.5 * g2n[g]) - math.log(g2n[g])

    write_group_stats('stats1.txt', groups, g2n, g2mean, g2sigmasq)
            
    # logger.info('scores %s', str(score_dict))   
    return score_dict

def scores_to_probs(score_dict):
    models = score_dict.keys()
    maxscore = np.max(score_dict.values())
    prob_dict = dict()
    for m in models:
        prob_dict[m] = math.exp(score_dict[m] - maxscore)
    prob_sum = np.sum(prob_dict.values())
    for m in models:
        prob_dict[m] = prob_dict[m] / prob_sum
    return(prob_dict)
    
def groups_to_probs(groups, g2n, g2mean, g2sigmasq):

    score_dict = dict()

    # select the groups
    my_n = [ g2n[g] for g in groups ]
    my_mean = [ g2mean[g] for g in groups ]
    my_sigmasq = [ g2sigmasq[g] for g in groups ]

    tot_n = np.sum(my_n)
    tot_sum = np.inner(my_n, my_mean)
    tot_mean = tot_sum / tot_n
    my_meansq = np.multiply(my_mean, my_mean)
    tot_sigmasqsum = np.inner(my_n, my_meansq + my_sigmasq) - ( tot_n * (tot_mean**2) )
    tot_sigmasq = tot_sigmasqsum / tot_n
    score_dict['zero'] = - 0.5 * tot_n * math.log(2.0 * math.pi * tot_sigmasq) - (0.5 * tot_n) - math.log(tot_n)

    share_sigmasq = np.inner( my_n, my_sigmasq ) / float(tot_n)
    # logger.info('tot_sigmasq %g share_sigmasq %g', tot_sigmasq, share_sigmasq)        
    score_dict['one'] = -0.5 * tot_n * math.log(2.0 * math.pi * share_sigmasq) - (0.5 * tot_n) - (0.5 * math.log(tot_n))
    for n in my_n:
        score_dict['one'] += -0.5 * math.log(float(n))

    score_dict['two'] = 0.0
    positive_sigmasq = [ s if s > 0 else share_sigmasq for s in my_sigmasq ]
    for (n, sigmasq) in zip(my_n, positive_sigmasq):
        n = float(n)
        score_dict['two'] += - 0.5 * n * math.log(2.0 * math.pi * sigmasq) - (0.5 * n) - math.log(n)

    prob_dict = scores_to_probs(score_dict)
    return(prob_dict, score_dict)
        
def compare_models(outdir, filename, group_vec, x_vec, name):
    
    sizemax = 16
    sizerange = range(1,sizemax+1)

    myfile = os.path.join(outdir, filename)
    plt.figure(figsize=(8,5))

    
    # for (x_vec, name, position) in [ (x1_vec, name1, 211), (x2_vec, name2, 212) ]:
        
    logger.info('comparing models for %s', name)

    model_probs = dict()
    
    for minsize in sizerange:

        logger.info('\n*** starting work on %s minsize = %d ***', name, minsize)
    
        score_dict = get_scores(group_vec, x_vec, minsize)
        best_score = max( score_dict.values() )
        deltascore_dict = dict()
        prob_dict = dict()
            
        prob_tot = 0.0
        for m in score_dict.keys():
            deltascore_dict[m] = score_dict[m] - best_score
            prob_dict[m] = math.exp(deltascore_dict[m])
            prob_tot += prob_dict[m]
        for m in models:
            prob_dict[m] = prob_dict[m] / prob_tot
        for m in models:
            model_probs[m] = model_probs.get(m, [ ]) + [ prob_dict[m] ]
        logger.info('*** Pr(M) for %s minsize %d ***', name, minsize)
        for m in models:
            logger.info('model %s\tscore %g\tdscore %g\tprob %g', m, score_dict[m], deltascore_dict[m], prob_dict[m])
    
     #   plt.subplot(position)
    plt.xlim(1,sizemax)
    plt.ylim(0,1)
    plt.plot(sizerange, model_probs['zero'], 'wo', label="Model 0")
    plt.plot(sizerange, model_probs['one'], 'k.-', label="Model 1")
    plt.plot(sizerange, model_probs['two'], 'k.--', label="Model 2")
    plt.xlabel('Smallest group size included')
    plt.ylabel('Model probability')
    plt.title(name)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)

    plt.savefig(myfile)
    plt.close()
    
    #logger.info('and now for get_loglik')
    #(llshare, loglik) = get_loglik(group_vec, x_vec, name)
    return None

def get_bootstrap_replicate(orig_vec):
    n = len(orig_vec)
    indices = nprand.randint(n, size=n) # return n integers uniform on [0, n)
    new_vec = [ orig_vec[i] for i in indices ]
    return(new_vec)

def bootstrap_models(outdir, plotfile, datafile, trials, lower, upper, group_vec, x_vec, name):
    
    logger.info('bootstrap models for %s, trials %d, lower %d, upper %d', name, trials, lower, upper)

    score_dict = get_scores(group_vec, x_vec, lower)
    best_score = max( score_dict.values() )
    deltascore_dict = dict()
    prob_dict = dict()
            
    prob_tot = 0.0
    for m in score_dict.keys():
        deltascore_dict[m] = score_dict[m] - best_score
        prob_dict[m] = math.exp(deltascore_dict[m])
        prob_tot += prob_dict[m]
    for m in prob_dict.keys():
        prob_dict[m] = prob_dict[m] / prob_tot
    logger.info('*** Pr(M) for %s minsize %d, no bootstrap ***', name, lower)
    for m in MODELS:
        logger.info('model %s\tscore %g\tdscore %g\tprob %g', m, score_dict[m], deltascore_dict[m], prob_dict[m])


    # pre-calculate group summary statistics to avoid recalculation
    (groups, g2n, g2mean, g2sigmasq) = get_group_stats(group_vec, x_vec)
    write_group_stats('stats2.txt', groups, g2n, g2mean, g2sigmasq)

    # calculate model probabilities for the original list
    (orig_prob_dict, orig_score_dict) = groups_to_probs(groups, g2n, g2mean, g2sigmasq)
    logger.info('original data scores: %f\t%f\t%f', orig_score_dict['zero'], orig_score_dict['one'], orig_score_dict['two'] )
    logger.info('original data probs: %f\t%f\t%f', orig_prob_dict['zero'], orig_prob_dict['one'], orig_prob_dict['two'] )

    threshold_list = [ ]
    nsubset_list = [ ]
    meansize_list = [ ]
    probzero_list = [ ]
    probone_list = [ ]
    probtwo_list = [ ]
    for minsize in range(lower, upper+1):
        subset = [ g for g in groups if (g2n[g] >= minsize) ]
        nsubset = len(subset)
        ntot = sum( [ g2n[g] for g in subset ] )
        meanval = float(ntot)/float(nsubset)
        logger.info('minsize %d has %d groups, %f observations per group', minsize, nsubset, meanval)
        sum_prob = dict()
        sum_size = 0.0
        for trial in range(trials):
            my_groups = get_bootstrap_replicate(subset)
            sum_size += sum( [ g2n[g] for g in my_groups ] )
            (prob_dict, score_dict) = groups_to_probs(my_groups, g2n, g2mean, g2sigmasq)
            for m in prob_dict.keys():
                sum_prob[m] = sum_prob.get(m, 0.0) + prob_dict[m]

        threshold_list.append(minsize)
        nsubset_list.append(nsubset)
        meansize_list.append( sum_size / float(nsubset * trials) )
        probzero_list.append( sum_prob['zero']/float(trials) )
        probone_list.append( sum_prob['one']/float(trials) )
        probtwo_list.append( sum_prob['two']/float(trials) )
        logger.info('minsize %d\ttrials %d\tnsubset %d\tmeansize %f\tzero %f\tone %f\ttwo %f',
                    minsize, trials, nsubset_list[-1], meansize_list[-1], probzero_list[-1], probone_list[-1], probtwo_list[-1])


    fp = open(os.path.join(outdir, datafile), 'w')
    fp.write('\t'.join([ 'threshold', 'trials', 'nsubset', 'meansize', 'probzero', 'probone', 'probtwo']) + '\n')
    for i in range(len(threshold_list)):
        fp.write('\t'.join( [ '%d' % threshold_list[i] , '%d' % trials , '%d' % nsubset_list[i],
                              '%f' % probzero_list[i] , '%f' % probone_list[i] , '%f' % probtwo_list[i] ] ) + '\n')
    fp.close()
    
    plt.figure(figsize=(8,15))
    plt.subplot(211)
    plt.xlabel('Minimum organoids per tumor')
    plt.ylabel('Number')
    plt.title('(A) Observations used for each bootstrap replicate') # get the comma separator
    plt.xlim(lower,upper)
    plt.ylim(0,max(meansize_list + nsubset_list))
    plt.plot(threshold_list, nsubset_list, 'k-', label='Number of tumors')
    plt.plot(threshold_list, meansize_list, 'k--', label='Number of organoids per tumor, bootstrap mean')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)

    plt.subplot(212)
    plt.xlabel('Minimum organoids per tumor')
    plt.ylabel('Model probability, bootstrap mean')
    plt.title('(B) Model probabilities, %s bootstraps' % '{:,}'.format(trials) )
    plt.xlim(lower-0.5,upper+0.5)
    plt.ylim(0,1)
    baralpha = 0.6
    # plot bars from top down so legend order agrees with bar order
    p2 = plt.bar(threshold_list, probtwo_list, bottom=[i+j for (i,j) in zip(probzero_list, probone_list)], align='center', color='blue', alpha=baralpha, label='Model 2')
    p1 = plt.bar(threshold_list, probone_list, bottom=probzero_list, align='center', color='green', alpha=baralpha, label='Model 1')
    p0 = plt.bar(threshold_list, probzero_list, color='red', align='center', alpha=baralpha, label='Model 0')
    plt.xticks(range(lower,upper+1))
    plt.yticks(np.linspace(0.0, 1.0, num=11))
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.savefig(os.path.join(outdir, plotfile))
    plt.close()
        
    return None


def write_boxplot(outdir, filename, axislabel, group_list, x_list):
    myfile = os.path.join(outdir, filename)
    logger.info('writing %s', myfile)

    group_to_data = dict()
    for (g, x) in zip(group_list, x_list):
        group_to_data[g] = group_to_data.get(g, [ ]) + [x]
        
    # the boxplot puts a red horizontal bar at the median
    # therefore, sort by median rather than mean
    group_to_mean = dict()
    group_to_median = dict()
    for g in group_to_data.keys():
        group_to_mean[g] = sum(group_to_data[g]) / float(len(group_to_data[g]))
        group_to_median[g] = np.median(group_to_data[g])
    # mg = [ (group_to_mean[g], g) for g in group_to_mean.keys()]
    mg = [ (group_to_median[g], g) for g in group_to_median.keys()]
    mg = sorted(mg)
    groups = [ g for (m, g) in mg ]
    #for g in groups:
    #    logger.info("group %s mean %f median %f", g, group_to_median[g], group_to_mean[g])
        
    data_list = [ group_to_data[g] for g in groups ]
    
    # logger.info('ctn_to_data %s', str(ctn_to_data))
    
    plt.figure(figsize=(20,8))
    plt.boxplot(data_list)
    indices = np.arange(1, 1+len(groups))
    ctn_short = [ str(int(g[3:])) for g in groups ]
    plt.xticks(indices, ctn_short)
    plt.ylabel(axislabel)
    plt.xlabel('Tumor')
    plt.savefig(myfile)
    plt.close()
    
    return None

def write_boxplot_pair(outdir, filename, axislabel1, axislabel2, group_list, x1_list, x2_list):
    myfile = os.path.join(outdir, filename)
    logger.info('writing %s', myfile)

    group_to_data = dict()
    for (g, x) in zip(group_list, x1_list):
        group_to_data[g] = group_to_data.get(g, [ ]) + [x]
        
    # the boxplot puts a red horizontal bar at the median
    # therefore, sort by median rather than mean
    group_to_median = dict()
    group_to_mean = dict()
    for g in group_to_data.keys():
        group_to_median[g] = np.median(group_to_data[g])
        group_to_mean[g] = np.mean(group_to_data[g])
    mg = [ (group_to_median[g], g) for g in group_to_median.keys()]
    mg = sorted(mg)
    groups = [ g for (m, g) in mg ]
    #for g in groups:
    #    logger.info("group %s mean %f median %f", g, group_to_mean[g], group_to_median[g])
        
    data_list = [ group_to_data[g] for g in groups ]
    
    # logger.info('ctn_to_data %s', str(ctn_to_data))
    
    mysize = 16
    biggersize = 20
    
    plt.figure(figsize=(20,16))

    plt.subplot(211)
    plt.boxplot(data_list)
    indices = np.arange(1, 1+len(groups))
    ctn_short = [ str(int(g[3:])) for g in groups ]
    plt.xticks(indices, ctn_short)
    # use one decimal point for yticks to match log10 plot
    (locs, labels) = plt.yticks()
    mylabels = [ ' %.1f' % x for x in locs ]
    plt.yticks(locs, mylabels)
    plt.title(axislabel1, fontsize=biggersize)
    plt.ylabel('Spectral power', fontsize=mysize)
    plt.xlabel('Tumor number', fontsize=mysize)
    
    # reuse the same name on purpose so it could go into a loop later
    group_to_data = dict()
    for (g, x) in zip(group_list, x2_list):
        group_to_data[g] = group_to_data.get(g, [ ]) + [x]
    data_list = [ group_to_data[g] for g in groups ]
    
    plt.subplot(212)
    plt.boxplot(data_list)
    indices = np.arange(1, 1+len(groups))
    ctn_short = [ str(int(g[3:])) for g in groups ]
    plt.xticks(indices, ctn_short)
    plt.title(axislabel2, fontsize=biggersize)
    plt.ylabel('log$_{10}$(Spectral power)', fontsize=mysize)
    plt.xlabel('Tumor number', fontsize=mysize)
    
    plt.savefig(myfile)
    plt.close()
    
    return None

def rsqbetween(log10tumor, log10organoid, zsq, wbratio):
    T = 10**log10tumor
    Nt = 10**log10organoid
    fac = zsq / (T - 1.0)
    val = (1.0 + (wbratio/Nt)) * fac / (1.0 + fac)
    return(val)

def rsqwithin(log10tumor, log10organoid, zsq, fracp):
    T = 10**log10tumor
    Nt = 10**log10organoid
    N = Nt * T
    fac = zsq / (fracp * (N - T - 1.0))
    val = fac / (1.0 + fac)
    return(val)
    
def write_powerplot(filename, type1err, type2err, wbratio, fracp):
    logger.info('type I %g type II %f wbratio %f fracp %f', type1err, type2err, wbratio, fracp)
    z1 = -norm_distribution.ppf(0.5 * type1err)
    z2 = norm_distribution.ppf(type2err)
    logger.info('z1 %f z2 %f', z1, z2)
    zsq = (z1 - z2)**2
    log10tumor = np.linspace(1.0, math.log10(200), 101)
    log10organoid = np.linspace(1.0, 3.0, 101)
    X, Y = np.meshgrid(log10tumor, log10organoid)
    Z = rsqbetween(X,Y,zsq,wbratio)
    logZ = np.log10(Z)

    plt.figure(figsize=(15,5))

    xtickvals = [10, 20, 50, 100, 200 ]
    xtickstrs = [ '{:d}'.format(int(v)) for v in xtickvals ]
    xticklocs = [ np.log10(v) for v in xtickvals ]
    ytickvals = [10, 20, 50, 100, 200, 500, 1000 ]
    ytickstrs = [ '{:d}'.format(int(v)) for v in ytickvals ]
    yticklocs = [ np.log10(v) for v in ytickvals ]

    ncontour = 100

    plt.subplot(121)
    plt.title(r'Between-Tumor Test, $\sigma_W^2/\sigma_B^2 = $' + str(wbratio))
    plt.xlabel('Number of Tumors')
    plt.ylabel('Number of Organoids per Tumor')
    plt.xticks(xticklocs, xtickvals)
    plt.yticks(yticklocs, ytickvals)
    
    plt.contourf(X,Y,logZ,ncontour)

    cbar = plt.colorbar()
    ctickvals = [ 1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1. ]
    cticklocs = np.log10(ctickvals)
    ctickstrs = ['{:f}'.format(v) for v in ctickvals ]
    #cbar.set_ticks(cticklocs)
    #cbar.set_ticklabels(ctickstrs)
    #cbar.ax.set_xlabel(r'$R^2_B$')

    plt.contour(X,Y,logZ,ncontour)

    plt.subplot(122)
    plt.title(r'Within-Tumor Test, $f_P = $' + str(fracp))
    plt.xlabel('Number of Tumors')
    plt.ylabel('Number of Organoids per Tumor')    
    plt.xticks(xticklocs, ytickvals)
    plt.yticks(yticklocs, ytickvals)
    
    Z = rsqwithin(X,Y,zsq,fracp)
    logZ = np.log10(Z)
    plt.contourf(X,Y,logZ,ncontour)

    cbar = plt.colorbar()
    ctickvals = [ 1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1. ]
    cticklocs = np.log10(ctickvals)
    ctickstrs = ['{:f}'.format(v) for v in ctickvals ]
    #cbar.set_ticks(cticklocs)
    #cbar.set_ticklabels(ctickstrs)
    #cbar.ax.set_xlabel(r'$R^2_W$')

    plt.contour(X,Y,logZ,ncontour)
    
    #lvls = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    #nfill = len(lvls) - 1
    #upvals = [ j/float(nfill - 1) for j in range(nfill) ]
    #downvals = upvals[:]
    #downvals.reverse()
    #zerovals = [ 0 for j in range(nfill)]
    #clrs = zip(zerovals, downvals, upvals)
    # clrs = ('green','palegreen','chartreuse','yellowgreen','yellow','orange','red')
    # logger.info('max z for between %g', np.max(Z))
    # plt.contourf(X,Y,Z, levels=lvls, colors=clrs)
    # mylvls = np.linspace(0,1.0,21)
    #mylvls = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #nfill = len(mylvls) - 1 # number of filled regions
    #nfill = 21
    #cmap = mpcm.get_cmap('terrain')
    #mycolors = tuple([ cmap(float(i)/float(nfill-1)) for i in range(nfill) ])
    #plt.contourf(X,Y,Z,20,norm=colors.LogNorm(vmin=0.001,vmax=1))
    

    plt.savefig(filename)
    plt.close()
    logger.info('power plots written to %s', filename)
    return None
    
def write_powerplot_arith(filename, type1err, type2err, wbratio, fracp):
    logger.info('type I %g type II %f wbratio %f fracp %f', type1err, type2err, wbratio, fracp)
    z1 = -norm_distribution.ppf(0.5 * type1err)
    z2 = norm_distribution.ppf(type2err)
    logger.info('z1 %f z2 %f', z1, z2)
    zsq = (z1 - z2)**2
    log10tumor = np.linspace(1.0, math.log10(200), 100)
    log10organoid = np.linspace(1.0, 3.0, 100)
    X, Y = np.meshgrid(log10tumor, log10organoid)
    #lvls = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    #nfill = len(lvls) - 1
    #upvals = [ j/float(nfill - 1) for j in range(nfill) ]
    #downvals = upvals[:]
    #downvals.reverse()
    #zerovals = [ 0 for j in range(nfill)]
    #clrs = zip(zerovals, downvals, upvals)
    # clrs = ('green','palegreen','chartreuse','yellowgreen','yellow','orange','red')
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.title(r'Between-Tumor Test, $\sigma_W^2/\sigma_B^2 = $' + str(wbratio))
    Z = rsqbetween(X,Y,zsq,wbratio)
    # logger.info('max z for between %g', np.max(Z))
    # plt.contourf(X,Y,Z, levels=lvls, colors=clrs)
    # mylvls = np.linspace(0,1.0,21)
    mylvls = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nfill = len(mylvls) - 1 # number of filled regions
    cmap = mpcm.get_cmap('terrain')
    mycolors = tuple([ cmap(float(i)/float(nfill-1)) for i in range(nfill) ])
    plt.contourf(X,Y,Z, levels=mylvls, colors=mycolors)
    #plt.contourf(X,Y,Z,20,norm=colors.LogNorm(vmin=0.001,vmax=1))
    xtickvals = [10, 20, 50, 100, 200 ]
    xticklocs = [ math.log10(v) for v in xtickvals ]
    ytickvals = [10, 20, 50, 100, 200, 500, 1000 ]
    yticklocs = [ math.log10(v) for v in ytickvals ]
    plt.xticks(xticklocs, xtickvals)
    plt.yticks(yticklocs, ytickvals)
    plt.xlabel('Number of Tumors')
    plt.ylabel('Number of Organoids per Tumor')
    cbar = plt.colorbar()
    cbar.ax.set_xlabel(r'$R^2_B$')
    plt.contour(X,Y,Z, levels=mylvls, colors=mycolors)
    
    plt.subplot(122)
    plt.title(r'Within-Tumor Test, $f_P = $' + str(fracp))
    Z = rsqwithin(X,Y,zsq,fracp)
    plt.contourf(X,Y,Z,levels=mylvls,colors=mycolors)
    plt.xticks(xticklocs, ytickvals)
    plt.yticks(yticklocs, ytickvals)
    plt.xlabel('Number of Tumors')
    plt.ylabel('Number of Organoids per Tumor')
    cbar = plt.colorbar()
    cbar.ax.set_xlabel(r'$R^2_W$')
    plt.contour(X,Y,Z,levels=mylvls,colors=mycolors)
    plt.savefig(filename)
    plt.close()
    return None

def read_organoid_wc(organoid_wc_file):
    fp = open(organoid_wc_file, 'r')
    fp.readline()
    pts = [ ]
    for line in fp:
        toks = line.strip().split()
        n = int(toks[0])
        pts.append(n)
    return pts

def make_histogram_pts_diam(outfile, pts_vec, diam_vec):

    plt.figure(figsize=(10, 4))
    plt.tight_layout()

    myaxes = plt.subplot(121)
    plt.title('(A) Manual boundary point distribution',fontsize='medium')
    plt.xlabel('Number of boundary points, manual segmentation')
    plt.ylabel('Number of organoids')
    xx = pts_vec
    # logger.info('%s mean %f std %f quartiles %s', field, mymean, mystd, str(myquartiles) )
    mywidth = 200
    nbins = math.ceil(max(xx)/float(mywidth))
    mybins = np.arange(0,mywidth*nbins, mywidth)
    plt.hist(xx, bins=mybins, fill=False)
    # plt.hist(xx, fill=False)
    plt.yscale('log', nonposy='clip')
    mymean = np.mean(xx)
    mystd = np.std(xx)
    myquartiles = np.percentile(xx, [25.,50.,75.])
    vals = tuple( [len(xx)] + [ np.around(x,0) for x in [mymean, mystd] + list(myquartiles)] )
    mytext = 'n = %d\n\nmean = %.0f\nstandard deviation = %.0f\n\n1st quartile = %.0f\nmedian = %.0f\n3rd quartile = %.0f' % vals
    plt.text(0.9,0.9,mytext,transform = myaxes.transAxes,ha='right',va='top')


    myaxes = plt.subplot(122)
    plt.title('(B) Effective diameter distribution',fontsize='medium')
    plt.xlabel('Effective diameter, $\mu m$')
    plt.ylabel('Number of organoids')
    xx = diam_vec
    mywidth = 40
    nbins = math.ceil(max(xx)/float(mywidth))
    mybins = np.arange(0,mywidth*nbins,mywidth)
    plt.hist(xx,bins=mybins,fill=False)
    # plt.hist(xx, fill=False)
    # plt.yscale('log', nonposy='clip')
    mymean = np.mean(xx)
    mystd = np.std(xx)
    myquartiles = np.percentile(xx, [25.,50.,75.])
    vals = tuple( [len(xx)] + [ np.around(x,0) for x in [mymean, mystd] + list(myquartiles)] )
    mytext = 'n = %d\n\nmean = %.0f\nstandard deviation = %.0f\n\n1st quartile = %.0f\nmedian = %.0f\n3rd quartile = %.0f' % vals
    plt.text(0.9,0.9,mytext,transform = myaxes.transAxes,ha='right',va='top')

    plt.savefig(outfile)

    return None

def make_histogram_k14(outfile, k14tot, k14mean):

    plt.figure(figsize=(10, 4))
    plt.tight_layout()

    mywidth = 0.05
    mymax = 1.0
    nbins = math.ceil(mymax/float(mywidth))
    mybins = np.arange(0,mywidth*nbins, mywidth)
    
    myaxes = plt.subplot(121)
    plt.title('(A) Total K14 per organoid',fontsize='medium')
    plt.xlabel('Totak K14')
    plt.ylabel('Number of organoids')
    xx = k14tot
    # logger.info('%s mean %f std %f quartiles %s', field, mymean, mystd, str(myquartiles) )
    plt.hist(xx, bins=mybins, fill=False)
    # plt.yscale('log', nonposy='clip')
    mymean = np.mean(xx)
    mystd = np.std(xx)
    myquartiles = np.percentile(xx, [25.,50.,75.])
    vals = tuple( [len(xx)] + [ np.around(x,3) for x in [mymean, mystd] + list(myquartiles)] )
    mytext = 'n = %d\n\nmean = %.3f\nstandard deviation = %.3f\n\n1st quartile = %.3f\nmedian = %.3f\n3rd quartile = %.3f' % vals
    plt.text(0.9,0.9,mytext,transform = myaxes.transAxes,ha='right',va='top')


    myaxes = plt.subplot(122)
    plt.title('(B) Mean K14 per organoid',fontsize='medium')
    plt.xlabel('Mean K14')
    plt.ylabel('Number of organoids')
    xx = k14mean
    plt.hist(xx,bins=mybins,fill=False)
    mymean = np.mean(xx)
    mystd = np.std(xx)
    myquartiles = np.percentile(xx, [25.,50.,75.])
    vals = tuple( [len(xx)] + [ np.around(x,3) for x in [mymean, mystd] + list(myquartiles)] )
    mytext = 'n = %d\n\nmean = %.3f\nstandard deviation = %.3f\n\n1st quartile = %.3f\nmedian = %.3f\n3rd quartile = %.3f' % vals
    plt.text(0.9,0.9,mytext,transform = myaxes.transAxes,ha='right',va='top')

    plt.savefig(outfile)

    return None

def main():
    
    parser = argparse.ArgumentParser(description='Invasive boundary image score, 2D',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--outdir', help='output directory', required=False, default='.')
    parser.add_argument('--organoid_table', help='organoid results', required=True)
    parser.add_argument('--organoid_wc', help='organoid wordcount', required=True)
    parser.add_argument('--bootstrap_trials', help='bootstrap number of replicates for model selection', type=int, required=False)
    parser.add_argument('--bootstrap_lower', help='bootstrap upper limit on group size', type=int, required=False, default=1)
    parser.add_argument('--bootstrap_upper', help='bootstrap upper limit on group size', type=int, required=False, default=10)    
    args = parser.parse_args()
    
    logger.info('outdir %s organoid_table %s', args.outdir, args.organoid_table)

    # check that the output directory exists; if not, create it
    if not os.path.isdir(args.outdir):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)

    # power figure moved to its own file
    # logger.info('skipping to powerplot and quitting')
    # write_powerplot(os.path.join(args.outdir, 'fig4power.pdf'), 0.05/20000., 0.2, 2.6, 0.8)
    # return None
    
    logger.info('reading saved results ...')
    organoid_table = read_table(os.path.join(args.outdir, args.organoid_table))

    organoids = sorted(organoid_table.keys())
    group_vec = [ organoid_table[o]['ctn'] for o in organoids ]

    pts_vec = read_organoid_wc(os.path.join(args.outdir, args.organoid_wc))
    area_total = 532.4 * 710.5 # area in square microns
    area_vec = [ area_total * float(organoid_table[o]['size_frac']) for o in organoids ]
    diam_vec = [ 2.0 * math.sqrt(a / math.pi) for a in area_vec ]

    logger.info('making histograms ...')
    histogram_pdf = os.path.join(args.outdir, 'fig1_histogram_pts_diam.pdf')
    make_histogram_pts_diam(histogram_pdf, pts_vec, diam_vec)

    k14_pdf = os.path.join(args.outdir, 'fig8_histogram_k14.pdf')
    k14tot = [ float(organoid_table[o]['k14_sum']) for o in organoids ]
    k14mean = [ float(organoid_table[o]['k14_mean']) for o in organoids ]
    make_histogram_k14(k14_pdf, k14tot, k14mean)

    
    for field in ('invasion_spectral', 'k14_sum', 'k14_mean', 'size_area', 'size_perimeter'):
        fname = field
        if field == 'invasion_spectral':
            fname = 'Invasion'
        x_vec = [ float(organoid_table[o][field]) for o in organoids ]
        minval = np.min(x_vec)
        if (minval <= 0):
            write_boxplot(args.outdir,
                        'boxplot_%s.pdf' % field,
                        '%s, arithmetic scale' % field,
                        group_vec, x_vec)            
            logger.info('%s: skipping log-scale boxplot because minval = %f', field, minval)
            continue
        lx_vec = [ math.log10(x) for x in x_vec ]
        write_boxplot_pair(args.outdir,
                           'boxplot_pair_%s.pdf' % fname,
                           '(A) %s, arithmetic scale' % fname,
                           '(B) %s, logarithmic scale' % fname,
                           group_vec, x_vec, lx_vec)


    y_vec = [ math.log10( float(organoid_table[o]['invasion_spectral']) ) for o in organoids ]
    for (field, xname, pdfname) in ( ('k14_sum', 'Total K14', 'fig9_k14total'),
                                     ('k14_mean', 'Mean K14', 'fig10_k14mean'),
                                     ('size_area', 'Organoid Area', 'fig11_area') ):
        x_vec = [ float(organoid_table[o][field]) for o in organoids ]
        x_vec = (rankdata(x_vec) - 0.5) / float(len(x_vec)) # rankdata gives ranks 1, 2, ..., n for n items
        yname = '$\log_{10}$' + ' Invasion' 
        tailfrac_line = np.linspace(0.5, 0.05, num=46)
        tailfrac_color = [ (0.5, 'darkblue'), (0.4, 'darkgreen'), (0.3, 'darkgoldenrod'), (0.2, 'darkorange'), (0.1, 'firebrick') ]
        fullpath = os.path.join(args.outdir, pdfname + '.pdf')
        test_scatter_pool(y_vec, x_vec, group_vec, yname, xname, tailfrac_line, tailfrac_color, fullpath)
        
    if (args.bootstrap_trials is not None):
        bootstrap_models(args.outdir, 'fig4_bootstrap.pdf', 'bootstrap_prob.txt',
                         args.bootstrap_trials, args.bootstrap_lower, args.bootstrap_upper, group_vec, lx_vec, 'Invasion, log10 scale')

    return None
    
if __name__ == "__main__":
    main()
