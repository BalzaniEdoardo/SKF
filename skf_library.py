import numpy as np
import scipy.stats as sts

def filterSKF(x_tm1_tm1, V_tm1_tm1, yt, F, H, Q, R,epsi=0):
    """
    Description:
    ============
        Filter step of the Switching Kalman Filter
    Input:
    ======
        :param x_tm1_tm1: mean of latents at time t - 1 given measures y until t-1
        :param V_tm1_tm1: cov of latents at time t-1 given measures y until t-1
        :param yt: measures at time t
        :param F: matrix of the linear dynamical system for the latents for a particular state
        :param H: matrix that generates the measures
        :param Q: noise covariance for the latents
        :param R: noise covariance for the measures
        :return:
    Output:
    =======
        :param x_t_t mean of latents at time t given measures y until t
        :param V_t_t cov of latents at time t given measures y until t
        :param V_t_tm1__t cov between latents at time t-1 and latents at time t, given measures y until time t
        :param L probability of yt given all measures until time t-1 and the current state
    """
    x_t_tm1 = np.dot(F,x_tm1_tm1)
    V_t_tm1 = np.dot(np.dot(F,V_tm1_tm1),F.T) + Q
    err = yt - np.dot(H,x_t_tm1)
    S = np.dot(np.dot(H,V_t_tm1),H.T) + R
    if np.sum(np.isnan(S)):
        print(S)
        pass
    Sinv = np.linalg.pinv(S+np.eye(S.shape[0])*epsi)

    K = np.dot(np.dot(V_t_tm1,H.T),Sinv)
    norm = sts.multivariate_normal(mean=np.zeros(err.shape),cov=S)
    L = norm.pdf(err)
    x_t_t = x_t_tm1 + np.dot(K,err)
    V_t_t = V_t_tm1 - np.dot(np.dot(K,S),K.T)
    deltaMat = np.eye(K.shape[0]) - np.dot(K,H)
    V_tm1_t__t = np.dot(np.dot(deltaMat,F),V_tm1_tm1)
    return x_t_t, V_t_t, V_tm1_t__t, L

def crossCollapseSKF(X,Y,cov,P):
    """
    Input:
    ======
    :param X: 2D array, (num of latents x num of states)
    :param Y: 2D array, (num of latents x num of states)
    :param cov: 3D array (num of latents x num of latents x num of states)
    :param P: 1D array, (num of states,)
    :return:
    """
    x_coll = np.dot(X,P)
    y_coll = np.dot(Y,P)
    dx = (X.T - x_coll).T
    dy = (Y.T - y_coll).T
    dxP = np.dot(dx,np.diag(P))
    cov_coll = np.dot(cov,P) + np.dot(dxP,dy.T)
    return x_coll, y_coll, cov_coll

def weightSumSKF(L,Z,M_tm1):
    M_ij = (np.dot((L*Z).T, np.diag(M_tm1.flatten()))).T
    M_ij = M_ij / np.sum(M_ij)
    M_t = np.sum(M_ij,axis=0)
    # print(1/M_t)
    W_t = np.dot(M_ij, np.diag(1/M_t))
    return W_t, M_t

if __name__ == '__main__':
    from scipy.io import savemat
    # set random seed
    np.random.seed(3)
    ## FIILTER AND SAVE THE RESULT

    # initialize paramentersd for filter
    x = np.random.normal(2,1,size=3)
    A = np.random.normal(0,1,size=(3,3))
    V = np.dot(A,A.T)
    y = np.random.normal(0,1,size=4)
    F = np.random.normal(0,1,size=(3,3))
    H = np.random.normal(0,1,size=(4,3))
    Q = np.eye(3)
    R = np.eye(4) + 0.2*np.diag(np.ones(3),-1) + 0.2*np.diag(np.ones(3),1)

    # perform a filter step and save
    x_t_t, V_t_t, V_tm1_t__t, L = filterSKF(x,V,y,F,H,Q,R)
    mdict = {'x':x.reshape(x.shape[0],1),'V':V,'y':y.reshape(y.shape[0],1),'F':F,'H':H,'Q':Q,'R':R,
             'x_t_t':x_t_t.reshape(x_t_t.shape[0],1), 'V_t_t':V_t_t, 'V_tm1_t__t':V_tm1_t__t, 'L':L}
    savemat('switchTest.mat',mdict=mdict)

    ## Check the collapse step
    # initialize the parameters
    stateNum = 2
    X = np.random.normal(size=(4,stateNum))
    Y = np.random.normal(size=(4,stateNum))
    P = np.random.uniform(0,1,size=stateNum-1)
    P = np.hstack((P,[1-sum(P)]))
    cov = np.zeros((4,4,stateNum))

    for j in range(stateNum):
        A = np.random.normal(size=(4, 4))
        cov[:,:,j] = np.dot(A,A.T)

    # perform a collapse
    x_coll, y_coll, cov_coll = crossCollapseSKF(X,X,cov,P)
    mdict = {'x_coll':x_coll.reshape(x_coll.shape[0],1),'y_coll':y_coll.reshape(y_coll.shape[0],1),'cov_coll':cov_coll,
             'X':X,'Y':Y,'P':P.reshape(P.shape[0],1),'cov':cov}
    savemat('collapseTest.mat', mdict=mdict)

    ## TEST WEIGHT
    L = np.random.normal(0,1,size=(2,2))
    Z = np.random.normal(0,1,size=(2,2))
    M_tm1 = np.random.normal(0,1,size=(2,1))
    W_t,M_t = weightSumSKF(L,Z,M_tm1)
    mdict = {'M_t':M_t,'W_t':W_t,'L':L,'Z':Z,'M_tm1':M_tm1}
    savemat('weightTest.mat', mdict=mdict)


