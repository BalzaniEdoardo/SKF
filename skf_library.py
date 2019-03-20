import numpy as np
import scipy.stats as sts
import control
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
# import scipy.linalg as sciLinalg

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


