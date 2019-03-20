import numpy as np
import sys
sys.path.append('/Users/edoardo/Work/Code/distributedsampling')
from LeapFrogOU import optimize_S_matrix
from scipy.spatial.distance import squareform
import scipy.stats as sts
import matplotlib.pyplot as plt

np.random.seed(3)
#### SAMPLER COV GENERATION
DIM = 2

# COV = np.diag(np.ones(DIM))
# for k in range(1,DIM):
#     COV = COV + (DIM-k)/50.*np.diag(np.ones(DIM-k),k)
# COV = COV + np.triu(COV,1).T
dt = 0.03

if DIM != 2:
#### N-DIM covariance creation
    eigVal = np.array([2.5,1,0.6,0.45,0.3])
    A = np.random.normal(size=(DIM,DIM))
    C = np.dot(A,A.T)
    e,H = np.linalg.eig(C)
    COV = np.dot(np.dot(H.T,np.diag(eigVal)),H)
    COVINV = np.linalg.pinv(COV)
    s2 = 1
    lam = 10**-16
    N = COV.shape[0]
    res = optimize_S_matrix(COV,COVINV,s2,N,lam,checkGrad=True)
    SM = squareform(res.x)
    SM = np.matrix(np.triu(SM) - np.tril(SM))
    OUMatrix = np.eye(DIM)+dt*np.dot(-s2*np.eye(COV.shape[0]) + SM, COVINV)#-np.dot(-s2*np.eye(COV.shape[0]) + SM, COVINV)
    OUMatrixNoRot = np.eye(DIM)+dt*10*np.dot(-s2*np.eye(COV.shape[0]), COVINV)

# 2 DIM change in correlation
else:
    print('2D version')
    theta = np.pi/4
    rotation = np.array(([np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]))
    D = np.diag([2.5,1])
    COV = np.dot(np.dot(rotation,D),rotation.T)

    theta = np.pi/4 + np.pi/2
    rotation = np.array(([np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]))
    COV2 = np.dot(np.dot(rotation,D),rotation.T)

    COVINV = np.linalg.pinv(COV)
    COVINV2 = np.linalg.pinv(COV2)

    ## AS a choice keep the same rotation!!!!!!
    s2 = 1
    lam = 10**-16
    N = COV.shape[0]
    res = optimize_S_matrix(COV2, COVINV2, s2, N, lam, checkGrad=True)
    SM = squareform(res.x)
    SM = np.matrix(np.triu(SM) - np.tril(SM))
    OUMatrixNoRot = np.eye(DIM) + dt * np.dot(-s2 * np.eye(COV2.shape[0]) + SM, COVINV2)
    OUMatrix = np.eye(DIM)+dt*np.dot(-s2*np.eye(COV.shape[0]) + SM, COVINV)#-np.dot(-s2*np.eye(COV.shape[0]) + SM, COVINV)
    print('S',res.x)


mu = np.zeros(DIM)
norm = sts.multivariate_normal(mean=mu, cov=COV)
x0 = norm.rvs()



## SAMPLER TRAJECTORY

sdt = np.sqrt(2)*np.sqrt(dt)
nIter = 50000
X = np.zeros((DIM,nIter+1))
RND = np.random.normal(loc = 0.0, scale = 1,size=(DIM,nIter))
for k in range(nIter):
    if k < nIter/2:
        X[:,k+1] =  np.dot(OUMatrix,X[:,k]) +  sdt* RND[:,k]
    else:

        X[:, k + 1] = np.dot(OUMatrixNoRot, X[:, k] ) + sdt * RND[:, k]

dTimes = 500
plt.plot(range(dTimes),X[0,nIter//2-dTimes:nIter//2],'r')
plt.plot(range(dTimes,2*dTimes),X[0,nIter//2:nIter//2+dTimes],'b')

DIM_y = 50
C = np.random.normal(size=(DIM_y,DIM))
R = 0.05*np.eye(DIM_y)
y = np.dot(C,X[:,1:]) + np.random.multivariate_normal(np.zeros(DIM_y),R,size=nIter).T

plt.figure()
plt.subplot(121)
plt.plot(range(dTimes),X[0,nIter//2-dTimes:nIter//2],'r')
plt.plot(range(dTimes,2*dTimes),X[0,nIter//2:nIter//2+dTimes],'b')
plt.subplot(122)
plt.plot(range(dTimes),y[0,nIter//2-dTimes:nIter//2],'r')
plt.plot(range(dTimes,2*dTimes),y[0,nIter//2:nIter//2+dTimes],'b')
plt.figure()
plt.plot(X[0,:nIter//2],X[1,:nIter//2],'r')
plt.plot(X[0,nIter//2:],X[1,nIter//2:],'b')

datDict = {
    'C':C,
    'y':y,
    'R':R,
    'x':X,
    'A1':OUMatrix,
    'A2':OUMatrixNoRot,
    'Q':np.eye(DIM)*sdt**2
}
np.save('exampleDynamic_%dD.npy'%(DIM),datDict)
from scipy.io import savemat
savemat('exampleDynamic_%dD.mat'%(DIM),datDict)