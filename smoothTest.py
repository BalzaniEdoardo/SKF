from skf_library import *
import numpy as np
import matplotlib.pyplot as plt

# import data

# datDict = np.load('oscill.npy').all()



datDict = np.load('exampleDynamic_2D_changeFreq.npy').all()
NState = 2;
M0 =  np.array([0.8, 0.2])
M_prev = M0.copy()
Q = np.array(datDict['Q']) # system noise
H = np.array(datDict['C']) # latent to measure proj
yt = np.array(datDict['y']) # measures
R = np.array(datDict['R']) # measurement noise
x = np.array(datDict['x']) # latent
A1 = np.array(datDict['A1'])
A2 = np.array(datDict['A2']) # latent linear dynamics


# initial conditions
# Z = np.array([[0.9999,0.0001],[0.0001,0.9999]])
Z = np.array([[0.999,0.001],[0.001,0.999]])
dictA = {0:A1,1:A2}
x0 = np.ones(x.shape[0])*0.01

cov0 = np.array(np.eye(x.shape[0]))

middle = yt.shape[1]//2

yt = yt[:,middle-10**4:middle+10**4]
x = x[:,middle-10**4:middle+10**4]
X_t_t, V_t_t, V_t_tm1__t, M_t_t = forwardPass(NState,x0,cov0,M0,yt,dictA,H,Q,R,Z)





x_t_T, V_t_T, M_t_T = backwardPassSKF(NState,X_t_t,V_t_t,V_t_tm1__t,M_t_t,dictA,Q,Z)
plt.figure()
middle = yt.shape[1]//2
# plt.plot(yt[0,middle-500:middle+500],'--',mfc='none',label='measure',alpha=0.3)
plt.plot(x[0,middle-500:middle+500],label='orig',alpha=0.8)
plt.plot(x_t_T[0,middle-500:middle+500],label='smoother')
plt.plot(X_t_t[0,0,middle-500:middle+500],label='filter')
plt.legend()

plt.figure()
plt.plot(M_t_T[0])
aa = np.ones(200)/200.
# smFilt = np.convolve(M_t_t[0],aa,mode='same')
plt.plot(M_t_t[0])
plt.plot([0,yt.shape[1]],[0.5,0.5],'--')

# detSmooth = np.zeros(V_t_T.shape[2])
# detFilt = np.zeros(V_t_T.shape[2])

# for k in range(detSmooth.shape[0]):
#     detSmooth[k] = np.linalg.det(V_t_T[:,:,k])
#     detFilt[k] = np.linalg.det(V_t_t[:,:,0,k])

# plt.figure()
# plt.plot(detFilt[1:]-detSmooth[1:])

plt.figure(figsize=[9.51, 8.31])
halfLen = yt.shape[1]//2
numDots = 500
ax = plt.subplot2grid((3, 2), (0, 0), colspan=2)
plt.title('Noisy measurement')
plt.plot(range(numDots),yt[0,halfLen-numDots:halfLen],'r',label='S=0')
plt.plot(range(numDots,2*numDots),yt[0,halfLen:halfLen+numDots],'b',label='S=1')
plt.legend(frameon=False)
ax = plt.subplot2grid((3, 2), (1, 0), colspan=2)
plt.plot(range(numDots),M_t_T[0,halfLen-numDots:halfLen],'r')
plt.plot(range(numDots,2*numDots),M_t_T[0,halfLen:halfLen+numDots],'b')
# plt.ylim(0.48,0.52)
plt.ylabel('Pr(S = 0)')
plt.plot([0,2*numDots],[0.5,0.5],'--k',lw=2)

ax = plt.subplot2grid((3, 2), (2, 0), colspan=2)
plt.plot(x[0,halfLen-numDots:halfLen+numDots],label='x')
plt.plot(x_t_T[0,halfLen-numDots:halfLen+numDots],label='$\\bar{x}$')
# plt.plot(x_hat[0,1,halfLen-numDots:halfLen+numDots],label='$\\bar{x}$|S=1')

plt.legend(frameon=False)
from scipy.io import savemat
savemat('err_t_T.mat',{'M_prevEdo':M_t_T})
plt.savefig('filteredTraectory_%d_covRotated_smooth.pdf'%yt.shape[0])

# plt.figure()
# plt.plot(range(0,halfLen),smoothProb[:halfLen],'b')
# plt.plot(range(halfLen,2*halfLen),smoothProb[halfLen+1:],'r')