#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:17:50 2018

@author: balzani
"""
from hamiltonianMCMC import *
import numpy as np
import matplotlib.pylab as plt
import control
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import scipy.linalg as sciLinalg
def autocorrelation (x) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size//2]/np.sum(xp**2) 

def lossFunction(S,Sigma,SigmaInv,s2,N,lam,isVector=True):
    """
    This function returns the value of phi_slow, using the relationship
    L(S) = trace(V P V^T)
    with V = $\\Lambda^{-1/2}$
    $\\Lambda = diag(\\Sigma)$
    S is skew symmetric matrix as vector
    """
    if isVector:
        SM = squareform(S)
        SM = np.matrix(np.triu(SM) - np.tril(SM))
    else:
        SM = S
    phi_slow,W = func_phislow(SM,Sigma,SigmaInv,s2,N)
    reg = (lam/(2*N**2)) * (np.linalg.norm(W, ord='fro')**2)
    
    L = phi_slow + reg
    return L

def autocorrNorm(t,S,Sigma,SigmaInv,s2,N):
    W = np.matrix(np.eye(S.shape[0])- s2*SigmaInv + S*SigmaInv)
    tmp = np.zeros((W.shape[0],W.shape[0],t.shape[0]))
    for k in range(t.shape[0]):
        tmp[:,:,k] = sciLinalg.expm(-(np.eye(W.shape[0]) - np.array(W) )*t[k])
    tmp = np.swapaxes(np.swapaxes(tmp,axis1=1,axis2=2),axis1=0,axis2=1)
    Kt = np.dot(tmp,np.array(Sigma))
    autoCorr = np.linalg.norm(Kt,ord='fro',axis=(1,2)) / np.linalg.norm(np.dot(np.ones(Sigma.shape),Sigma), ord='fro')
    return autoCorr
    

def func_phislow(S,Sigma,SigmaInv,s2,N):
    W = np.matrix(np.eye(S.shape[0])- s2*SigmaInv + S*SigmaInv)
    Lambda = np.matrix(np.diag(np.diag(Sigma)))
    C = symmetrize(Sigma * np.linalg.pinv(Lambda) * Sigma)
    P = np.matrix(control.lyap(W-np.eye(W.shape[0]), C))
    sqrtLambdaInv = np.matrix(np.sqrt(np.linalg.pinv(Lambda)))
    return np.trace(sqrtLambdaInv*P*sqrtLambdaInv)/(2*N**2),W

def symmetrize(M):
    if np.max(np.abs(M-M.T)) > 10**-15:
        raise ValueError("Matrix M must be symmetric up to numerical error")
    if type(M) == np.matrix:
        M = np.triu(M)
        M = M + np.triu(M,1).T
        M = np.matrix(M)
    else:
        M = np.triu(M)
        M = M + np.triu(M,1).T
    return M

def grad_phislow(S,Sigma,SigmaInv,s2,N):
    W = np.eye(S.shape[0])- s2*SigmaInv + S*SigmaInv
    Lambda = np.diag(np.diag(Sigma))
    C = symmetrize(Sigma * np.linalg.pinv(Lambda) * Sigma)
    P = np.matrix(control.lyap(W-np.eye(W.shape[0]), C))
    Q = np.matrix(control.lyap((W-np.eye(W.shape[0])).T, np.linalg.pinv(Lambda)))
    return ((SigmaInv*P*Q).T - (SigmaInv*P*Q))/(2*N**2)

def getMatricesVU(Sigma):
    if type(Sigma) != np.matrix:
        raise TypeError('Sigma must be a np.matrix')
    V = np.matrix(np.diag(np.sqrt(1/np.diagonal(Sigma))))
    U = Sigma*V
    return V,U


def grad_loss(S,Sigma,SigmaInv,s2,N,lam,isVector=True):
    if isVector:
        SM = squareform(S)
        SM = np.matrix(np.triu(SM) - np.tril(SM))
    else:
        SM = S
    
    
    phi_grad = grad_phislow(SM,Sigma,SigmaInv,s2,N)
    
    regGrad = lam * (SM*(SigmaInv*SigmaInv) + (SigmaInv*SigmaInv)*SM) / (2*N**2)
    
    grad = np.triu(regGrad + phi_grad,1)
    grad = grad + grad.T

    return squareform(grad)

def grad_check(dS,SM,Sigma,SigmaInv,s2,N,lam,isVector=False):
    if isVector:
        dS = squareform(dS)
        dS = np.matrix(np.triu(dS) - np.tril(dS))
        SM = squareform(SM)
        SM = np.matrix(np.triu(SM) - np.tril(SM))
        
    dSApprox = np.zeros(SM.shape[0]*(SM.shape[0]-1)//2)
    count = 0
    for row in range(SM.shape[0]):
        for col in range(row+1,SM.shape[0]):
            Stepmat = np.zeros(SM.shape)
            Stepmat[row,col] = 10**(-5) / np.sqrt(2)
            Stepmat[col,row] = -10**(-5) / np.sqrt(2)
            dSApprox[count] = (lossFunction(SM+Stepmat,Sigma,SigmaInv,s2,N,lam,isVector=False)-lossFunction(SM-Stepmat,Sigma,SigmaInv,s2,N,lam,isVector=False)) / (2*10**-5)
            count+=1
    dSExact = rotateGrad(dS)
    return dSApprox, dSExact

def rotateGrad(dS):
    vecDs = np.zeros(dS.shape[0]*(dS.shape[0]-1)//2)
    dS = np.matrix(dS)
    count = 0
    for row in range(dS.shape[0]):
        for col in range(row+1,dS.shape[0]):
            v = np.matrix(np.zeros(dS.shape))
            v[row,col] = 1/np.sqrt(2)
            v[col,row] = -1/np.sqrt(2)
            vecDs[count] = np.trace(dS*v.T) 
            count += 1
    return vecDs

def optimize_S_matrix(Sigma,SigmaInv,s2,N,lam,checkGrad=False):
    
    # initialize S
    S = np.random.choice([-1,1],size=(N*(N-1)//2,))*np.random.uniform(0,0.11,size=(N*(N-1)//2,))

    if checkGrad:
        dS = grad_loss(S,Sigma,SigmaInv,s2,N,lam,isVector=True)
        dSa,dSe = grad_check(dS,S,Sigma,SigmaInv,s2,N,lam,isVector=True)
#        print('Grad check: ', np.max(np.abs(dSa-dSe)) )
        
    # define function
    func = lambda S :lossFunction(S,Sigma,SigmaInv,s2,N,lam,isVector=True)
    
    # define gradient
    grad_func = lambda S : grad_loss(S,Sigma,SigmaInv,s2,N,lam,isVector=True)
    
    res = minimize(func, S, method='L-BFGS-B', jac=grad_func, tol=10**-15)
    
    return res
    
def eulerMaruyama(x0,nIter,M,mu,deltat):
    x0 = x0.flatten()
    xtmp = np.zeros((x0.shape[0],nIter))
    drift = np.dot(np.linalg.pinv(M),mu)
    
    for k in range(nIter):
        xtmp[:,k] = x0 + deltat*( - np.dot(M,x0) + drift) + np.random.normal(loc = 0.0, scale = np.sqrt(deltat),size=x0.shape)
        x0 = xtmp[:,k]
    return xtmp

def rungeKuttaSDE(x0,nIter,M,mu,deltat,nGauss=2**10,taum=1.):
    x0 = x0.flatten()
    xtmp = np.zeros((x0.shape[0],nIter))
    drift = np.dot(M,mu)
    
    for k in range(nIter):
        if k % nGauss == 0:
            Wsize = min(k+nGauss,nIter-k)
            dW = np.random.normal(loc = 0.0, scale = np.sqrt(deltat),size=(x0.shape[0],Wsize))
            countGauss = 0
#        dW = np.random.normal(loc = 0.0, scale = np.sqrt(dt),size=x0.shape)
        K1 = deltat * (- np.dot(M,x0) + drift)/taum + dW[:,countGauss]/np.sqrt(taum)
        K2 = deltat * (- np.dot(M,x0+K1) + drift)/taum + dW[:,countGauss]/np.sqrt(taum)
        xtmp[:,k] = x0 + 0.5*(K1+K2)
        countGauss += 1
        x0 = xtmp[:,k]
    return xtmp


if __name__ == '__main__':
    plt.close('all')
    N=2
    S = np.random.choice([-1,1],size=(N*(N-1)//2,))*np.random.uniform(0,0.01,size=(N*(N-1)//2,))
    

    s2 = 1.
    lam = 0.00000001
    Sigma = np.matrix([[0.12,0.09],[0.09,0.12]])*0.001
    SigmaInv = np.matrix(np.linalg.pinv(Sigma))
    
    res = optimize_S_matrix(Sigma,SigmaInv,s2,N,lam,checkGrad=True)
    
#    optimalS = res.x
    d = {'Sigma':Sigma,'s2':s2,'Sopt':res.x,'lam':lam}
#    np.save('/Local/Users/balzani/Code/DistributedSampling/Parameters/Optimized_S_N_%d.npy'%N,d)
#    
#    # OU SIMULATION
#    saveSim = False
#    x0 = np.ones((2,1))
#    x0[:,0] = [-5.,-5.]
##    x0 = x0.flatten()
#    p0 = np.ones((2,1))
#    p0[:,0] = [-1,1]
#    mu = np.ones(x0.shape[0])*1.
#    
#    eta = 0.01
#   
#    nIter = 10**5
#    burnIn = 10**4
#    checkAcceptance = False
#    L = 1
#    
#    
#    
#    covKinetic = s2*np.diag(np.ones(x0.shape[0]))
#    kinetic = lambda x : gaussianPotential(x, np.zeros(x0.shape[0]), covKinetic)
#    gradKinetic = lambda x : gradGaussianPotential(x, np.zeros(x0.shape[0]), covKinetic)
#    
#    acceptanceRule = HMCMCAcceptance
#
##    # SIGNLE GAUSSIAN EXAMPLE
##    np.random.seed(3)
#    SM = squareform(res.x)
#    SM = np.matrix(np.triu(SM) - np.tril(SM))
#    A = 0.5*(-np.dot(-s2*np.eye(Sigma.shape[0]) + SM,np.linalg.pinv(Sigma)))
#    
#    potential = lambda x : gaussianPotential(x, mu, np.array(Sigma))
#    gradPotential = lambda x : gradGaussianPotentialSigmaInv(x, mu, np.array(A))
    
## =============================================================================
##     HMC for generating path es external input
## =============================================================================
#    eig = np.linalg.eig(A)[0]
#    R = np.diag(eig)
#    taum=0.02
#    
#    x2 = eulerMaruyama(x0,nIter,np.array(A),mu,eta)
#    x1,p1 = hamiltonianIntegration(x0, nIter, potential, gradPotential, kinetic, gradKinetic,
#                             covKinetic, eta, L, acceptanceRule=HMCMCAcceptance, useRule=checkAcceptance)
#    
    
#    gradPotential = lambda x : gradGaussianPotentialSigmaInv(x, mu, np.array(np.linalg.pinv(Sigma)))
#    x2,p2 = hamiltonianIntegration(x0, nIter, potential, gradPotential, kinetic, gradKinetic,
#                             covKinetic, eta, L, acceptanceRule=HMCMCAcceptance, useRule=checkAcceptance)
#    
#    x1 = x1[:,burnIn:]
#    p1 = p1[:,burnIn:]
#    w, v = np.linalg.eigh(Sigma[:2,:2]) #eigenvalues and eigenvectors
#    t = np.linspace(0,np.pi*2,100) # equispaced points for ellipses plot
#    Chi2 = sts.chi2(2)
#    scaled_w = np.array([np.cos(t),np.sin(t)]).T * np.sqrt(w* Chi2.ppf(0.99)) 
#    ellip = np.dot(scaled_w,v) + mu[:2]
#    plt.figure()
#    plt.title('Hamiltonian MCMC: %d steps, stepsize = %.2f'%(L,eta))
#    plt.plot(x1[0,burnIn:],x1[1,burnIn:],'ob')
#    plt.plot(ellip[:,0],ellip[:,1],'r')
#    plt.plot(x1[0,:35],x1[1,:35],'-oy')
#   
#    plt.figure()
#    plt.hist(x1[0,:],bins=40,alpha=0.2,label='OU',normed=True)
#    plt.hist(x2[0,:],bins=40,alpha=0.2,label='Langevin',normed=True)
#    plt.title('Sampling comparrison using leapfrog')
#    plt.legend()
#    plt.figure()
#    acf1 = autocorrelation(x1[1,burnIn:])
#    acf2 = autocorrelation(x2[1,burnIn:])
#    plt.plot(acf1,label='Optimized OU')
#    plt.plot(acf2,label='Langevin')
#    plt.legend()
#    plt.title('Autocorrelation comparrison - numerical')
#    
#    t=np.linspace(0,10,100)
#    acfOpt = autocorrNorm(t,SM,Sigma,SigmaInv,s2,N)
#    acfLan = autocorrNorm(t,np.zeros(SM.shape[0]),Sigma,SigmaInv,s2,N)
#    S = np.random.choice([-1,1],size=(N*(N-1)//2,))*np.random.uniform(0,0.5,size=(N*(N-1)//2,))
#    S = squareform(S)
#    S = np.matrix(np.triu(S) - np.tril(S))
#    acfRand = autocorrNorm(t,S,Sigma,SigmaInv,s2,N)
#    plt.figure()
#    plt.title('Theoretical Acf')
#    plt.plot(t,acfOpt,label='Optimized OU')
#    plt.plot(t,acfLan,label='Langevin')
#    plt.plot(t,acfRand,ls='--',label='Random Skewsymmetric')
#    plt.legend()
#    plt.xlabel('Time(AU)')
#    plt.ylabel('Normalised ACF')
#    
##    
##    descr = "The x1[:,0] is the initial condition, x[:,k+1] is the position\
##             corresponding to the momentum p[:,k]"
##    di = {'p':p1,'x':x1,'eta':eta,'descr':descr,'mu':mu,'Sigma':Sigma}
##    np.save('/Local/Users/balzani/Code/theory_sparse_representation/hmc_results_0,002.npy',di)
