__author__ = 'vestrada'

import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import grizli
import matplotlib.image as mpimg
from astropy.io import fits
#from vtl.Readfile import Readfile
from astropy.io import ascii
from astropy.table import Table
import cPickle
import os
from glob import glob
from time import time
import seaborn as sea

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
R = robjects.r
pandas2ri.activate()

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({"xtick.direction": "in", "ytick.direction": "in"})
colmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.1, as_cmap=True)


def Oldest_galaxy(z):
    return cosmo.age(z).value


def Gauss_dist(x, mu, sigma):
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    C = np.trapz(G, x)
    G /= C
    return G


def NormalP(dependent, p):
    ncoeff = np.trapz(p, dependent)
    p /= ncoeff
    return p


def Median_w_Error(Pofx, x):
    iP = interp1d(x, Pofx)
    ix = np.linspace(x[0], x[-1], 500)

    lerr = 0
    herr = 0

    for i in range(len(ix)):
        e = np.trapz(iP(ix[0:i + 1]), ix[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = ix[i]
        if herr == 0:
            if e >= .84:
                herr = ix[i]
                break

    med = 0

    for i in range(len(x)):
        e = np.trapz(Pofx[0:i + 1], x[0:i + 1])
        if med == 0:
            if e >= .5:
                med = x[i]
                break

    return np.round(med,3), np.round(med - lerr,3), np.round(herr - med,3)

def Median_w_Error_95(Pofx, x):
    iP = interp1d(x, Pofx)
    ix = np.linspace(x[0], x[-1], 500)

    lerr = 0
    herr = 0

    for i in range(len(ix)):
        e = np.trapz(iP(ix[0:i + 1]), ix[0:i + 1])
        if lerr == 0:
            if e >= .025:
                lerr = ix[i]
        if herr == 0:
            if e >= .975:
                herr = ix[i]
                break

    med = 0

    for i in range(len(x)):
        e = np.trapz(Pofx[0:i + 1], x[0:i + 1])
        if med == 0:
            if e >= .5:
                med = x[i]
                break

    return np.round(med,3), np.round(med - lerr,3), np.round(herr - med,3)

def Median_w_Error_cont(Pofx, x):
    ix = np.linspace(x[0], x[-1], 500)
    iP = interp1d(x, Pofx)(ix)

    C = np.trapz(iP,ix)

    iP/=C


    lerr = 0
    herr = 0
    med = 0

    for i in range(len(ix)):
        e = np.trapz(iP[0:i + 1], ix[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = ix[i]
        if med == 0:
            if e >= .50:
                med = ix[i]
        if herr == 0:
            if e >= .84:
                herr = ix[i]
                break

    return med, med - lerr, herr - np.abs(med)


def Scale_model(D, sig, M):
    C = np.sum(((D * M) / sig ** 2)) / np.sum((M ** 2 / sig ** 2))
    return C


def Identify_stack(fl, err, mfl):
    x = ((fl - mfl) / err) ** 2
    chi = np.sum(x)
    return chi


def Likelihood_contours(age, metallicty, prob):
    ####### Create fine resolution ages and metallicities
    ####### to integrate over
    m2 = np.linspace(min(metallicty), max(metallicty), 50)

    ####### Interpolate prob
    P2 = interp2d(metallicty, age, prob)(m2, age)

    ####### Create array from highest value of P2 to 0
    pbin = np.linspace(0, np.max(P2), 1000)
    pbin = pbin[::-1]

    ####### 2d integrate to find the 1 and 2 sigma values
    prob_int = np.zeros(len(pbin))

    for i in range(len(pbin)):
        p = np.array(P2)
        p[p <= pbin[i]] = 0
        prob_int[i] = np.trapz(np.trapz(p, m2, axis=1), age)

    ######## Identify 1 and 2 sigma values
    onesig = np.abs(np.array(prob_int) - 0.68)
    twosig = np.abs(np.array(prob_int) - 0.95)

    return pbin[np.argmin(twosig)],pbin[np.argmin(onesig)] 



"""Test Functions"""


def Best_fit_model(input_file, metal, age, tau):
    dat = fits.open(input_file)

    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    x = np.argwhere(chi == np.min(chi))
    print metal[x[0][0]], age[x[0][1]], tau[x[0][2]]
    return metal[x[0][0]], age[x[0][1]], tau[x[0][2]]


#####JWST FIT

def Analyze_JWST_LH(chifits, specz, metal, age, tau, age_conv='../Quiescent_analysis/data/tau_scale_nirspec.dat'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    scale = np.loadtxt(age_conv,skiprows=1).T

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    ######## Reshape likelihood to get average age instead of age when marginalized
    newchi = np.zeros(chi.shape)

    for i in range(len(chi)):
        if i == 0:
            newchi[i] = chi[i]
        else:
            frame = interp2d(metal, scale[i], chi[i])(metal, age[:-overhead[i]])
            newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

    ####### Create normalize probablity marginalized over tau
    P = np.exp(-newchi.T.astype(np.float128) / 2)

    prob = np.trapz(P, ultau, axis=2)
    C = np.trapz(np.trapz(prob, age, axis=1), metal)

    prob /= C

    #### Get Z and t posteriors

    PZ = np.trapz(prob, age, axis=1)
    Pt = np.trapz(prob.T, metal,axis=1)

    return prob.T, PZ,Pt


def Nirspec_fit(sim_spec, filters,specz, metal, age, tau, name, inc_err=1):
    #############Read in spectra#################
    wv, fl, er = np.load(sim_spec)
    er *= inc_err
    
    flx = np.zeros(len(wv))

    for i in range(len(wv)):
        if er[i] > 0:
            flx[i] = fl[i] + np.random.normal(0, er[i])
    #############Prep output files###############
    chifile = 'chidat/%s_JWST_chidata' % name

    ##############Create chigrid and add to file#################
    mflx = np.zeros([len(metal)*len(age)*len(tau),len(wv)])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                mwv, mfl = np.load('../Quiescent_analysis/JWST/m%s_a%s_t%s_%s.npy' %
                                   (metal[i], age[ii], tau[iii],filters))
                mfl *=(mwv)**2 / 3E18
                C = Scale_model(flx,er,mfl)
                mflx[i*len(age)*len(tau)+ii*len(tau)+iii]=mfl*C
    chigrid = np.sum(((flx - mflx) / er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile, chigrid)

    P, PZ, Pt = Analyze_JWST_LH(chifile + '.npy', specz, metal, age, tau)

    np.save('chidat/%s_tZ_pos' % name,P)
    np.save('chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('chidat/%s_t_pos' % name,[age,Pt])
    np.save('output/nirspec_sim_data_%s' % name,[wv,fl,flx,er])

    print 'Done!'
    return



def Highest_likelihood_model_JWST(spec, filters, bfmetal, bfage, tau):
    wv, fl, flx, er = np.load(spec)
    fp = '../JWST/'

    chi = []
    for i in range(len(tau)):
        mwv, mfl = np.load(fp + 'm%s_a%s_t%s_%s.npy' % (bfmetal, bfage, tau[i], filters))
        mfl *= (mwv) ** 2 / 3E14
        C = Scale_model(flx, er, mfl)
        chi.append(Identify_stack(fl, er, C * mfl))

    return bfmetal, bfage, tau[np.argmin(chi)]


def MC_fit_jwst(sim_spec, metal, age, tau, name, repeats=100, age_conv='../data/tau_scale_nirspec.dat'):
    ####### Get maximum age
    max_age = Oldest_galaxy(3.717)

    mlist = []
    alist = []

    ultau = np.append(0, np.power(10, np.array(tau[1:]) - 9))
    iZ = np.linspace(metal[0], metal[-1], 100)
    it = np.linspace(age[0], age[-1], 100)

    wv, fl, er = np.load(sim_spec)
    fl = fl [wv<4.9]
    er = er [wv<4.9]

    ###############Get model list
    mflx = np.zeros([len(metal)*len(age)*len(tau),len(wv[wv<4.9])])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                mwv, mfl = np.load('../JWST/m%s_a%s_t%s_nirspec.npy' %
                                   (metal[i], age[ii], tau[iii]))
                C = Scale_model(fl,er,mfl[wv<4.9])
                mflx[i*len(age)*len(tau)+ii*len(tau)+iii]=mfl[wv<4.9]*C

    scale = Readfile(age_conv)

    overhead = np.zeros(len(scale)).astype(int)
    for i in range(len(scale)):
        amt = []
        for ii in range(len(age)):
            if age[ii] > scale[i][-1]:
                amt.append(1)
        overhead[i] = sum(amt)

    for xx in range(repeats):
        flx = fl + np.random.normal(0, er)

        chi = np.sum(((flx - mflx) / er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).astype(np.float128)

        chi[:, len(age[age <= max_age]):, :] = 1E5
        ######## Reshape likelihood to get average age instead of age when marginalized
        newchi = np.zeros(chi.T.shape)

        for i in range(len(newchi)):
            if i == 0:
                newchi[i] = chi.T[i]
            else:
                frame = interp2d(metal, scale[i], chi.T[i])(metal, age[:-overhead[i]])
                newchi[i] = np.append(frame, np.repeat([np.repeat(1E5, len(metal))], overhead[i], axis=0), axis=0)

        ####### Create normalize probablity marginalized over tau
        prob = np.exp(-newchi.T.astype(np.float128) / 2)

        P = np.trapz(prob, ultau, axis=2)
        C = np.trapz(np.trapz(P, age, axis=1), metal)

        #### Get Z and t posteriors
        PZ = np.trapz(P / C, age, axis=1)
        Pt = np.trapz(P.T / C, metal, axis=1)

        iPZ = interp1d(metal, PZ)(iZ)
        iPt = interp1d(age, Pt)(it)

        med = 0
        for i in range(len(iZ)):
            e = np.trapz(iPZ[0:i + 1], iZ[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = iZ[i]

        mlist.append(med)

        med = 0
        for i in range(len(it)):
            e = np.trapz(iPt[0:i + 1], it[0:i + 1])
            if med == 0:
                if e >= .5:
                    med = it[i]

        alist.append(med)

    np.save('../mcerr/' + name, [mlist, alist])

    return


#####STATS

def Leave_one_out(dist, x):
    Y = np.zeros(x.size)
    for i in range(len(dist)):
        Y += dist[i]
    Y /= np.trapz(Y, x)

    w = np.arange(.01, 2.01, .01)
    weights = np.zeros(len(dist))
    for i in range(len(dist)):
        Ybar = np.zeros(x.size)
        for ii in range(len(dist)):
            if i != ii:
                Ybar += dist[ii]
        Ybar /= np.trapz(Ybar, x)
        weights[i] = np.sum((Ybar - Y) ** 2) ** -1
    return weights

def Stack_posteriors(P_grid, x):
    P_grid = np.array(P_grid)
    W = Leave_one_out(P_grid,x)
    top = np.zeros(P_grid.shape)
    for i in range(W.size):
        top[i] = W[i] * P_grid[i]
    P =sum(top)/sum(W)
    return P / np.trapz(P,x)

def Iterative_stacking(grid_o,x_o,iterations = 20,resampling = 250):
    ksmooth = importr('KernSmooth')
    del_x = x_o[1] - x_o[0]

    ### resample
    x = np.linspace(x_o[0],x_o[-1],resampling)
    grid = np.zeros([len(grid_o),x.size])    
    for i in range(len(grid_o)):
        grid[i] = interp1d(x_o,grid_o[i])(x)
   
    ### select bandwidth
    H = ksmooth.dpik(x)
    ### stack posteriors w/ weights
    stkpos = Stack_posteriors(grid,x)
    ### initialize prior as flat
    Fx = np.ones(stkpos.size)
    
    for i in range(iterations):
        fnew = Fx * stkpos / np.trapz(Fx * stkpos,x)
        fx = ksmooth.locpoly(x,fnew,bandwidth = H)
        X = np.array(fx[0])
        iFX = np.array(fx[1])
        Fx = interp1d(X,iFX)(x)

    Fx[Fx<0]=0
    Fx = Fx/np.trapz(Fx,x)
    return Fx,x

def Linear_fit(x,Y,sig,new_x):
    A=np.array([np.ones(len(x)),x]).T
    C =np.diag(sig**2)
    iC=inv(C)
    b,m = np.dot(inv(np.dot(np.dot(A.T,iC),A)),np.dot(np.dot(A.T,iC),Y))
    cov = inv(np.dot(np.dot(A.T,iC),A))
    var_b = cov[0][0]
    var_m = cov[1][1]
    sig_mb = cov[0][1]
    sig_y = np.sqrt(var_b + new_x**2*var_m + 2*new_x*sig_mb)
    return m*new_x+b , sig_y

def Bootstrap_errors_lfit(masses,metals,ers,sampling=np.arange(10,11.75,.01),its=1000):
    l_grid = np.zeros([its,len(sampling)])
    IDs = np.arange(len(masses))
    for i in range(its):
        IDn = np.random.choice(IDs,len(IDs),replace=True)
        lvals = np.polyfit(masses[IDn],np.log10(metals[IDn]/.019),1,w = 1/ers[IDn]**2)
        lfit = np.polyval(lvals,sampling)
        l_grid[i] = lfit
        
    m_fit = np.mean(l_grid,axis=0)
    low_ers = np.zeros(len(samp))
    hi_ers = np.zeros(len(samp))
    
    for i in range(len(l_grid.T)):
        low_ers[i] = np.sort(l_grid.T[i])[150]
        hi_ers[i] = np.sort(l_grid.T[i])[830]
    return low_ers,hi_ers, m_fit

nirspec_box = [[[14,20,38.36], [52,57,11.8]],[[14,20,38.33], [52,58,47.0]],[[14,20,28.95], [52,58,46.7]],[[14,20,28.99], [52,57,12.5]],
[[14,20,38.28], [52,59,11.7]], [[14,20,38.20], [53, 0,46.8]], [[14,20,28.92], [53, 0,46.5]], [[14,20,28.84], [52,59,11.4]],
[[14,20,24.77], [52,59,10.4]], [[14,20,24.79], [53, 0,45.6]], [[14,20,15.62], [53, 0,43.2]], [[14,20,15.55], [52,59,10.0]],
[[14,20,24.91], [52,57,13.6]], [[14,20,24.65], [52,58,46.7]], [[14,20,15.66], [52,58,45.3]], [[14,20,15.65], [52,57,11.7]],
[[14,20,21.12], [52,57,31.3]], [[14,20,21.02], [52,59,05.9]], [[14,20,11.64], [52,59,06.0]], [[14,20,11.80], [52,57,29.3]],
[[14,20,21.03], [52,59,28.1]], [[14,20,21.10], [53, 1,04.2]], [[14,20,11.54], [53, 1,04.3]], [[14,20,11.70], [52,59,29.7]],
[[14,20,07.51], [52,59,29.7]], [[14,20,07.41], [53, 1,03.4]], [[14,19,58.18], [53, 1,01.4]], [[14,19,58.46], [52,59,27.7]],
[[14,20,07.55], [52,57,28.9]], [[14,20,07.62], [52,59,05.0]], [[14,19,58.46], [52,59,04.6]], [[14,19,58.40], [52,57,29.9]],
[[14,20,16.12], [52,53,25.6]], [[14,20,16.13], [52,55,01.8]], [[14,20,06.65], [52,55,00.3]], [[14,20,06.81], [52,53,25.2]],
[[14,20,16.13], [52,55,24.9]], [[14,20,16.09], [52,56,59.6]], [[14,20,06.66], [52,56,58.2]], [[14,20,06.76], [52,55,23.0]],
[[14,20,02.41], [52,55,23.5]], [[14,20,02.47], [52,56,58.7]], [[14,19,53.27], [52,56,56.2]], [[14,19,53.43], [52,55,22.5]],
[[14,20,02.74], [52,53,23.7]], [[14,20,02.52], [52,54,59.3]], [[14,19,53.32], [52,54,59.9]], [[14,19,53.55], [52,53,25.2]],
[[14,19,58.84], [52,53,44.8]], [[14,19,58.84], [52,55,20.0]], [[14,19,49.36], [52,55,21.0]], [[14,19,49.59], [52,53,44.3]],
[[14,19,58.96], [52,55,44.2]], [[14,19,58.79], [52,57,19.8]], [[14,19,49.25], [52,57,19.3]], [[14,19,49.47], [52,55,42.6]],
[[14,19,45.29], [52,55,44.6]], [[14,19,45.34], [52,57,19.3]], [[14,19,36.02], [52,57,16.7]], [[14,19,35.97], [52,55,43.5]],
[[14,19,45.57], [52,53,44.8]], [[14,19,45.40], [52,55,19.5]], [[14,19,36.26], [52,55,19.9]], [[14,19,36.32], [52,53,45.2]],
[[14,19,54.10], [52,49,42.2]], [[14,19,54.05], [52,51,17.3]], [[14,19,44.58], [52,51,16.3]], [[14,19,44.75], [52,49,41.1]],
[[14,19,53.99], [52,51,41.0]], [[14,19,53.88], [52,53,16.1]], [[14,19,44.57], [52,53,15.1]], [[14,19,44.58], [52,51,41.4]],
[[14,19,40.46], [52,51,39.9]], [[14,19,40.39], [52,53,15.0]], [[14,19,31.20], [52,53,12.9]], [[14,19,31.38], [52,51,38.3]],
[[14,19,40.69], [52,49,41.1]], [[14,19,40.52], [52,51,15.2]], [[14,19,31.38], [52,51,14.6]], [[14,19,31.45], [52,49,40.5]],
[[14,19,36.85], [52,49,55.1]], [[14,19,36.56], [52,51,30.8]], [[14,19,27.20], [52,51,29.1]], [[14,19,27.39], [52,49,54.0]],
[[14,19,36.56], [52,51,52.9]], [[14,19,36.49], [52,53,30.1]], [[14,19,27.13], [52,53,28.0]], [[14,19,27.31], [52,51,52.8]],
[[14,19,23.08], [52,51,53.2]], [[14,19,23.00], [52,53,28.4]], [[14,19,13.86], [52,53,25.2]], [[14,19,13.94], [52,51,51.5]],
[[14,19,23.21], [52,49,53.4]], [[14,19,23.25], [52,51,30.1]], [[14,19,14.06], [52,51,28.9]], [[14,19,14.14], [52,49,53.2]]]