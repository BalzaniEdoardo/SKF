import numpy as np
import scipy.stats as sts
import control
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
# import scipy.linalg as sciLinalg

def smoothSKF(x_tp1_T,V_tp1_T,x_t_t,V_t_t,V_tp1_tp1,V_tp1_t__tp1,F,Q):
    """
    Input
    =====
            :x_tp1_T:
               .. math:: x_{t+1|T}^{k} = E[X_{t+1} | y_{1:T}\, ; S_{t+1} = k]
            :V_tp1_T:
                .. math:: V_{t+1|T}^{k} = Cov[X_{t+1} | y_{1:T}\, ; S_{t+1} = k]
            :x_t_t:
                .. math:: x_{t|t}^{j} = E[X_t | y_{1:t}\, ; S_t = j]
            :V_t_t:  
                .. math:: V_{t|t}^{j} = Cov[X_t | y_{1:t}\, ; S_t = j]
            :V_tp1_tp1:
                .. math::  V_{t+1|t+1}^{k} = Cov[X_{t+1} | y_{1:t+1}\, ; S_{t+1} = k]
            :V_tp1_t__tp1:
                .. math:: Cov[X_{t+1}, X_{t} | y_{1:t+1}\,; S_{t}=j\,; S_{t+1}=k]
            :F: Linear dynamics for the latent variables
            :Q: System noise covariance matrix
        
    """
    
    x_tp1_t = np.dot(F,x_t_t)
    V_tp1_t = np.dot(np.dot(F,V_t_t),F.T) + Q

    # gain
    J = np.dot(V_t_t, np.dot(F.T, np.linalg.pinv(V_tp1_t)))

    x_t_T = x_t_t + np.dot(J, x_tp1_T - x_tp1_t)
    V_t_T = V_t_t + np.dot(np.dot(J,V_tp1_T - V_tp1_t),J.T)
    V_tp1_t__T = V_tp1_t__tp1 + np.dot(np.dot(V_tp1_T-V_tp1_tp1, np.linalg.pinv(V_tp1_tp1)),V_tp1_t__tp1)
    return x_t_T,V_t_T,V_tp1_t__T

def smoothStateProb(M_t_t,Z,M_tp1_T):
    U = (Z.T * M_t_t).T / np.dot(M_t_t,Z)
    # Utmp = np.zeros(Z.shape)
    # for j in range( M_t_t.shape[0]):
    #     Utmp[j,:] = M_t_t[j] * Z[j,:]/np.dot(M_t_t,Z)
    # print('delta U')
    # print(U-Utmp)

    M_t_tp1__T = U * M_tp1_T

    # Mtmp = np.zeros(Z.shape)
    # for k in range(Z.shape[0]):
    #     Mtmp[:,k] = U[:,k]*M_tp1_T[k]
    #
    # print('delta M',M_t_tp1__T-Mtmp)

    M_t_T = np.sum(M_t_tp1__T,1)

    # Mtmp2 = np.dot(U,M_tp1_T)
    # print('deltaM 2')
    # print(Mtmp2-M_t_T)

    W = M_t_tp1__T.T / M_t_T
    # Wtmp = np.zeros(Z.shape)
    # for j in range(Z.shape[0]):
    #     Wtmp[:,j] = M_t_tp1__T[j,:]/M_t_T[j]
    # print('delta W')
    # print(W-Wtmp)
    return M_t_T, W, U

# def smoothSKFEasy(x_tp1_T,V_tp1_T,x_t_t,V_t_t,V_tp1_tp1,V_tp1_t__tp1,F,Q):
#     x_tp1_T = np.matrix(x_tp1_T).T
#     V_tp1_T = np.matrix(V_tp1_T)
#     x_t_t = np.matrix(x_t_t).T
#     V_t_t = np.matrix(V_t_t)
#     V_tp1_tp1 = np.matrix(V_tp1_tp1)
#     V_tp1_t__tp1 = np.matrix(V_tp1_t__tp1)
#     F = np.matrix(F)
#     Q = np.matrix(Q)
#
#     # Smooth step
#     x_tp1_t = F*x_t_t
#     V_tp1_t = F*V_t_t*F.T + Q
#     J = V_t_t * F.T * np.linalg.pinv(V_tp1_t)
#
#     x_t_T = x_t_t + J * (x_tp1_T - x_tp1_t)
#     V_t_T = V_t_t + J * (V_tp1_T - V_tp1_t) * J.T
#     V_tp1_t__T = V_tp1_t__tp1 + (V_tp1_T - V_tp1_tp1) * np.linalg.pinv(V_tp1_tp1) * V_tp1_t__tp1
#     return x_t_T, V_t_T, V_tp1_t__T

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
    :X: 
        2D array, (num of latents x num of states)
    :Y: 
        2D array, (num of latents x num of states)
    :cov: 
        3D array (num of latents x num of latents x num of states)
    :P: 
        1D array, (num of states,)
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

def forwardPass(NState,x0,V0,M0,yt,dictA,H,Q,R,Z):
    """
    Input:
    =====
        :NState: number of hidden states
        :x0: m x 1 initial condition for the latent mean
        :V0: m x m initial condition for the latent covariance
        :yt: N x T observations
        :dictA: dicitionary with the different linear dynamics for the hidden (keys are the state)
        :H: N x m projection matrix from hidden to observable
        :Q: m x m system error covariance matrix
        :R: N x N observation error covariance matrix
    Output:
    =======
        :x_t_t:
            .. math::
                x_{t|t}^{j} = E[X_t | y_{1:t}\, ; S_t = j]
            Array containing the mean of the latent at time t, given the measure up to time t  and a state at t
        :V_t_t:
            .. math::
                V_{t|t}^{j} = Cov[X_t | y_{1:t}\, ; S_t = j]
            Array containing the covariance of the latent at time t, given the measure up to time t and a state at t
        :V_t_tm1__t:
            .. math::
                Cov[X_{t}, X_{t-1} | y_{1:t}\,; S_{t-1}=j\,; S_{t}=k]
        :M_t_t:
            .. math::
                M_{t|t}^{j} = P(S_t = j | y_{1:t})
            Vector of the probabilities of being in each state given the measure up to t
    """
    X_t_t = np.zeros((x0.shape[0], NState, yt.shape[1]+1))
    V_t_t = np.zeros((*V0.shape, NState, yt.shape[1]+1))
    M_t_t = np.zeros((NState, yt.shape[1]+1))
    V_t_tm1__t = np.zeros((*V0.shape,NState,NState,yt.shape[1]+1))

    # set initial conditions
    for j in  range(NState):
        X_t_t[:,j,0] = x0
        V_t_t[:,:,j,0] = V0
        M_t_t[j,0] = M0[j]

    # define containers that will be used in the loop

    xtmp = np.zeros((x0.shape[0], NState))
    Vtmp = np.zeros((*V0.shape,NState))
    Lmat = np.zeros((NState,NState))

    # loop over time
    for tt in range(yt.shape[1]):
        print(tt,yt.shape[1])
        xDict = {}
        vDict = {}

        for j in range(NState):
            for i in range(NState):
                xPrev = X_t_t[:, i, tt]
                vPrev = V_t_t[:, :, i, tt]

                xtmp[:, i], Vtmp[:, :, i], V_t_tm1__t[:,:,i,j,tt+1], Lmat[i, j] = \
                    filterSKF(xPrev, vPrev, yt[:, tt], dictA[j], H, Q, R)
            xDict[j] = xtmp.copy()
            vDict[j] = Vtmp.copy()

        W, M_t_t[:, tt+1] = weightSumSKF(Lmat, Z, M_t_t[:, tt])
        for i in range(NState):
            X_t_t[:, i, tt + 1], _, V_t_t[:, :, i, tt + 1] =\
                crossCollapseSKF(xDict[i], xDict[i], vDict[i], W[:, i])

    return X_t_t, V_t_t,V_t_tm1__t, M_t_t


def oneStepSmooth(NState, x_tp1_T, V_tp1_T, x_t_t, V_t_t, V_tp1_tp1, VV_tp1_tp1, dictA, Q, M_t_t, M_tp1_T, Z):
    """
    Descrition
    ==========
        Function implementing a single step backward of the Switching Kalman smooter.
        The equation for input and output given here follow the notation of Murphy K. 1998.
    Input
    =====
        :NState: number of hidden dynamics
        :x_tp1_T:
            .. math::
                x_{t+1|T}^{k} = E[X_{t+1} | y_{1:T}\, ; S_{t+1} = k]

            Array containing the mean of the latent at time t+1 given all the measures and a fixed state at t+1
        :V_tp1_T:
            .. math::
                V_{t+1|T}^{k} = Cov[X_{t+1} | y_{1:T}\, ; S_{t+1} = k]
            Array containing the covariance of the latent at time t+1 given all the measures and a fixed state at t+1
        :x_t_t:
            .. math::
                x_{t|t}^{j} = E[X_t | y_{1:t}\, ; S_t = j]
            Array containing the mean of the latent at time t, given the measure up to time t  and a state at t
        :V_t_t:
            .. math::
                V_{t|t}^{j} = Cov[X_t | y_{1:t}\, ; S_t = j]
            Array containing the covariance of the latent at time t, given the measure up to time t and a state at t
        :V_tp1_tp1:
            .. math::
                V_{t+1|t+1}^{k} = Cov[X_{t+1} | y_{1:t+1}\, ; S_{t+1} = k]
            Array containing the covariance of the latent at time t+1, given the measure up to time t+1 and a state at t+1
        :VV_tp1_tp1:
            .. math::
                Cov[X_{t+1}, X_{t} | y_{1:t+1}\,; S_{t}=j\,; S_{t+1}=k]
        :dictA:
            dict, keys are the tate indexes, values are the linear dynamics for the latent variables
        :Q:
            system noise covariance
        :M_t_t:
            .. math::
                M_{t|t}^{j} = P(S_t = j | y_{1:t})
            Vector of the probabilities of being in each state given the measure up to t
        :M_tp1_T:
            .. math::
                M_{t+1|T}^{j} = P(S_{t+1} = j | y_{1:T})
            Array of the prob of being in a state at t+1 given all measures
        :Z:
            Markov transition matrix, prior probability of switching from a state to  another

    Output
    ======
        :x_t_T:
            .. math::
                x_{t|T}^{j} = E[X_t | y_{1:T}, S_t=j]
            Array containing the mean of the latent given a state and all measures, one backward step given the input
        :V_t_T:
            .. math::
                V_{t|T}^{k} = Cov[X_t | y_{1:T}\, ; S_{t} = k]
            Array containing the covariance of the latent given a state and all measures, one backward step given the input
        :M_t_T:
            .. math::
                M_{t+1|T}^{j} = P(S_{t+1} = j | y_{1:T})
            Array of the prob of being in a state at t given all measures
    """
    M_t_T, W, U = smoothStateProb(M_t_t, Z, M_tp1_T)

    x_t_T = np.zeros((x_tp1_T.shape[0], NState))
    V_t_T = np.zeros((V_t_t.shape))
    XJK_t_T = np.zeros((x_tp1_T.shape[0], NState,NState)) # contain all

    for j in range(NState):

        xtmp = np.zeros((x_tp1_T.shape[0], NState))
        Vtmp = np.zeros(V_t_t.shape)

        for k in range(NState):
            xjk_t_T, Vjk_t_T, Vjk_tp1_t__T = smoothSKF(x_tp1_T[:, k], V_tp1_T[:, :, k],
                                                       x_t_t[:, j], V_t_t[:, :, j],
                                                       V_tp1_tp1[:, :, k],
                                                       VV_tp1_tp1[:, :, j, k], dictA[k], Q)
            XJK_t_T[:,j,k] = xjk_t_T.copy()
            xtmp[:, k] = xjk_t_T.copy()
            Vtmp[:, :, k] = Vjk_t_T.copy()

        x_t_T[:, j], _, V_t_T[:, :, j] = crossCollapseSKF(xtmp, xtmp, Vtmp, W[:, j])

    xx = np.zeros(x)
    crossCollapseSKF()
    return x_t_T, V_t_T, M_t_T

def backwardPassSKF(NState,X_t_t,V_t_t,V_t_tm1__t,M_t_t,dictA,Q,Z):

    # initialize loop
    x_t_T = np.zeros((X_t_t.shape[0],X_t_t.shape[2]))
    V_t_T = np.zeros((V_t_t.shape[0],V_t_t.shape[1],X_t_t.shape[2]))

    xj_t_T = np.zeros(X_t_t.shape)
    Vj_t_T = np.zeros(V_t_t.shape)
    M_t_T = np.zeros(M_t_t.shape)

    xj_t_T[:,:,-1] = X_t_t[:,:,-1]
    Vj_t_T[:,:,:,-1] = V_t_t[:,:,:,-1]
    M_t_T[:,-1] = M_t_t[:,-1]

    x_t_T[:,-1],_,V_t_T[:,:,-1] = crossCollapseSKF(xj_t_T[:, :, -1],xj_t_T[:, :, -1],Vj_t_T[:, :, :, -1], M_t_T[:, -1] )

    for tt in range(X_t_t.shape[-1]-2,-1,-1):
        print(tt,X_t_t.shape[-1])
        x_t_t_ = X_t_t[:,:,tt]
        V_t_t_ = V_t_t[:,:,:,tt]
        V_tp1_tp1 = V_t_t[:,:,:,tt+1]
        VV_tp1_tp1 = V_t_tm1__t[:,:,:,:, tt+1]
        M_t_t_ = M_t_t[:,tt]
        A,B,C = oneStepSmooth(NState, xj_t_T[:,:,tt+1],Vj_t_T[:,:,:,tt+1],x_t_t_,V_t_t_,
                              V_tp1_tp1,VV_tp1_tp1,dictA,Q,M_t_t_,M_t_T[:,tt+1],Z)

        xj_t_T[:, :, tt] = A
        Vj_t_T[:, :, :, tt] = B
        M_t_T[:, tt] = C
        x_t_T[:,tt],_, V_t_T[:,:,tt] = crossCollapseSKF(xj_t_T[:, :, tt],xj_t_T[:, :, tt],Vj_t_T[:, :, :, tt], M_t_T[:, tt] )
    return x_t_T, V_t_T, M_t_T

def func_phislow(S,Sigma,SigmaInv,s2,N):
    W = np.matrix(np.eye(S.shape[0])- s2*SigmaInv + S*SigmaInv)
    Lambda = np.matrix(np.diag(np.diag(Sigma)))
    C = symmetrize(Sigma * np.linalg.pinv(Lambda) * Sigma)
    P = np.matrix(control.lyap(W-np.eye(W.shape[0]), C))
    sqrtLambdaInv = np.matrix(np.sqrt(np.linalg.pinv(Lambda)))
    return np.trace(sqrtLambdaInv*P*sqrtLambdaInv)/(2*N**2),W

def symmetrize(M):
#    if np.max(np.abs(M-M.T)) > 10**-13:
#        raise ValueError("Matrix M must be symmetric up to numerical error")
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
    
    res = minimize(func, S, method='L-BFGS-B', jac=grad_func, tol=10**-10)
    
    return res

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


