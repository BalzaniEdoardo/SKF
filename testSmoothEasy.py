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


x = np.array(datDict['x']) # latent
A1 = np.array(datDict['A1'])
A2 = np.array(datDict['A2']) # latent linear dynamics

H = np.eye(2)
R = np.eye(2) * .5

yt = np.dot(H,x) + np.random.multivariate_normal(mean=np.zeros(2),cov=R,size=x.shape[1]).T

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
plt.plot(yt[0,middle-50:middle+50],'--',mfc='none',label='measure',alpha=0.3)
plt.plot(x[0,middle-50:middle+50],label='orig',alpha=0.8)
plt.plot(x_t_T[0,middle-50:middle+50],label='smoother')
plt.plot(X_t_t[0,0,middle-50:middle+50],label='filter')


plt.legend()

plt.figure()
plt.plot(range(0,M_t_T.shape[1]//2),M_t_T[0,:M_t_T.shape[1]//2],'b')
plt.plot(range(M_t_T.shape[1]//2,M_t_T.shape[1]),M_t_T[0,M_t_T.shape[1]//2:],'r')
aa = np.ones(200)/200.
# smFilt = np.convolve(M_t_t[0],aa,mode='same')
# plt.plot(M_t_t[0])
plt.plot([0,yt.shape[1]],[0.5,0.5],'--')

detSmooth = np.zeros(V_t_T.shape[2])
detFilt = np.zeros(V_t_T.shape[2])

# for k in range(detSmooth.shape[0]):
#     detSmooth[k] = np.linalg.det(V_t_T[:,:,k])
#     detFilt[k] = np.linalg.det(V_t_t[:,:,0,k])
#
# plt.figure()
# plt.plot(detFilt[1:]-detSmooth[1:])

