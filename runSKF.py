from skf_library import *
import numpy as np
import matplotlib.pyplot as plt

# import data

# datDict = np.load('oscill.npy').all()
datDict = np.load('exampleDynamic_2D.npy').all()
NState = 2;
M_prev = np.array([0.5, 0.5])
Q = np.array(datDict['Q']) # system noise
H = np.array(datDict['C']) # latent to measure proj
yt = np.array(datDict['y']) # measures
R = np.array(datDict['R']) # measurement noise
x = np.array(datDict['x']) # latent
A1 = np.array(datDict['A1'])
A2 = np.array(datDict['A2']) # latent linear dynamics

# initial conditions
Z = np.array([[0.999,0.001],[0.001,0.999]])
dictA = {0:A1,1:A2}
x_prev = np.zeros((x.shape[0],NState))
x_prev[:,0] = np.ones(x.shape[0])*0.01#[0,0]
x_prev[:,1] = np.zeros(x.shape[0])#[0.1,-0.2]
cov0 = np.array(np.eye(x.shape[0]))
v_prev = np.zeros(((x.shape[0],x.shape[0],NState)))
v_prev[:,:,0] = cov0 #* 0.002
v_prev[:,:,1] = cov0 #* 0.001


# SKF run
np.random.seed(3)
# container for approx and probabilities for each state
x_hat = np.zeros((x.shape[0],NState,x.shape[1]))
M_hat = np.zeros((NState,x.shape[1]))

xtmp = np.zeros((x.shape[0], NState))
Vtmp = np.zeros(v_prev.shape)
Lmat = np.zeros((NState,NState))


for tt in range(yt.shape[1]):
    print( tt)
    xDict = {}
    vDict = {}
    for j in range(NState):


        for i in range(NState):
            xPrev = x_prev[:,i]
            vPrev = v_prev[:,:,i]

            x_t_t, V_t_t, V_tm1_t__t, L = filterSKF(xPrev, vPrev, yt[:,tt], dictA[j], H, Q, R)
            Lmat[i,j] = L
            xtmp[:,i] = x_t_t
            Vtmp[:,:,i] = V_t_t
        xDict[j] = xtmp.copy()
        vDict[j] = Vtmp.copy()
    W, M_prev = weightSumSKF(Lmat, Z, M_prev)
    M_hat[:,tt] = M_prev
    # print(W)
    for i in range(NState):
        x_prev[:,i],_, v_prev[:,:,i] = crossCollapseSKF(xDict[i],xDict[i], vDict[i], W[:, i])
        x_hat[:, i, tt] = x_prev[:,i]

plt.figure(figsize=[9.51, 8.31])
halfLen = yt.shape[1]//2
numDots = 500
ax = plt.subplot2grid((3, 2), (0, 0), colspan=2)
plt.title('Noisy measurement')
plt.plot(range(numDots),yt[0,halfLen-numDots:halfLen],'r',label='S=0')
plt.plot(range(numDots,2*numDots),yt[0,halfLen:halfLen+numDots],'b',label='S=1')
plt.legend(frameon=False)
ax = plt.subplot2grid((3, 2), (1, 0), colspan=2)
smoothProb = np.convolve(M_hat[0,:],np.ones(100)/100.,mode='valid')
plt.plot(range(smoothProb.shape[0]//2),smoothProb[:smoothProb.shape[0]//2],'r')
plt.plot(range(smoothProb.shape[0]//2,smoothProb.shape[0]),smoothProb[smoothProb.shape[0]//2:],'b')
# plt.ylim(0.48,0.52)
plt.ylabel('Pr(S = 0)')
plt.plot([0,2*halfLen],[0.5,0.5],'--k',lw=2)

ax = plt.subplot2grid((3, 2), (2, 0), colspan=2)
plt.plot(x[0,halfLen-1000:halfLen+1000],label='original latent')
plt.plot(x_hat[0,0,halfLen-1000:halfLen+1000],label='$\\bar{x}$|S=0')
plt.plot(x_hat[0,1,halfLen-1000:halfLen+1000],label='$\\bar{x}$|S=1')

plt.legend(frameon=False)
from scipy.io import savemat
savemat('err.mat',{'M_prevEdo':M_hat})
plt.savefig('filteredTraectory_%d_covRotated.pdf'%yt.shape[0])