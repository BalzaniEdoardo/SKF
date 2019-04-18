#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:41:03 2019

@author: edoardo
"""

from skf_library import *
from scipy.io import loadmat
import numpy as np
for z in [1,20]:#'0,010','0,100',5,10,15,20]:
    try:
        param = loadmat('/Users/edoardo/Work/Code/sampling_in_gsm/optimParam_%d.mat'%z)
    except:
        param = loadmat('/Users/edoardo/Work/Code/sampling_in_gsm/optimParam_%s.mat' % z)
    lam = param['lam'][0][0]
    lamUsed = lam * 0
    s2 = param['s2'][0][0]
    S0 = param['S0'].flatten()
    N = param['N'][0,0]
    dS = param['dS'].flatten()
    #dSa = param['dSa'].flatten()
    #dSe = param['dSe'].flatten()
    Sigma = param['Sigma']
    SigmaInv = param['SigmaInv']

    Sigma = np.matrix(Sigma)
    SigmaInv = np.matrix(SigmaInv)
    #dS1 = grad_loss(S0,Sigma,SigmaInv,s2,N,lam)
    #[dSa1,dSe1] = grad_check(dS,S0,Sigma,SigmaInv,s2,N,lam,isVector=True)
    #
    res = optimize_S_matrix(Sigma,SigmaInv,s2,N,lam)
    #print('grad1 vs grad2')
    #print(max(np.abs(dS-dS1)))
    #print('grad checking python')
    #print(max(np.abs(dSa1-dSe1)))
    #print('grad approx python vs matlab')
    #print(max(np.abs(dSa1-dSa)))
    #print('grad exact python vs matlab')
    #print(max(np.abs(dSe1-dSe)))

    #print(res)
    #print(grad_loss(res.x,Sigma,SigmaInv,s2,N,lam))
    #print(lossFunction(res.x,Sigma,SigmaInv,s2,N,lam,isVector=True),lossFunction(np.zeros(res.x.shape),Sigma,SigmaInv,s2,N,lam,isVector=True))
    if lamUsed == 0:
        try:
            np.save('/Users/edoardo/Work/Code/sampling_in_gsm/optimizedS_Full_model_%d_noReg.npy'%z,{'res':res,'param':param,'lamUsed':lamUsed})
        except:
            np.save('/Users/edoardo/Work/Code/sampling_in_gsm/optimizedS_Full_model_%s_noReg.npy' % z,
                    {'res': res, 'param': param, 'lamUsed': lamUsed})
    elif lamUsed < lam:
        try:
            np.save('/Users/edoardo/Work/Code/sampling_in_gsm/optimizedS_Full_model_%d_smallLam.npy'%z,{'res':res,'param':param,'lamUsed':lamUsed})
        except:
            np.save('/Users/edoardo/Work/Code/sampling_in_gsm/optimizedS_Full_model_%s_smallLam.npy' % z,
                    {'res': res, 'param': param, 'lamUsed': lamUsed})