import numpy as np
import matplotlib.pylab as plt

dt = 0.001
a = 0.1 # m/s**2
A1 = np.matrix([[1,  dt, 0.5*a*dt**2],
      [0,  1,  dt],
      [0,  0,  1]])

a = -0.1
A2 = np.matrix([[1,  dt, 0.5*dt**2],
                [0,  1,  dt],
                [0,  0,  1]])



dictA = {0:A1,1:A2}
x0 = np.array([0,0,a]).reshape(1,3)



nIter = 15*100**3


xt = np.zeros((3,nIter))
xt[:,0] = x0
useDyn = 0
A = A1
for k in range(1,nIter):
    if k % 3000 == 0:
        useDyn += 1
        useDyn = useDyn % 2
        A = dictA[useDyn]
    xx = xt[:,k-1].reshape(3,1)
    xt[:,k] = (A * xx).flatten()

plt.figure()
plt.subplot(311)
plt.plot(xt[0,:])
plt.subplot(312)
plt.plot(xt[1,:])
plt.subplot(313)
plt.plot(xt[2,:])