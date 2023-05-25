import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import linear_kernel
from dual_quad_matrix import dual_quad_matrix


def FnormA_pr_perm(X,y,n_classes):
    # requires: cvxopt
    # FnormA stands for Frobenius norm with preconditioner A
    
    k = n_classes
    d = X.shape[1]
    n = len(y)
    
    am = get_AMPLE_matrices(k)
    XcheckT_list = [np.array(((X[i,:].reshape(-1,1)@
                               np.eye(k-1)[j,:].reshape(1,-1))@
                              am[y[i]]).flatten(order = 'F')) 
                    for i in range(n) 
                    for j in range(k-1)]
    XcheckT = np.vstack(XcheckT_list)
    
    ones_column = np.ones((k-1, 1))
    ident = np.eye(k-1)
    R = np.hstack((ones_column, -ident))
    A = np.linalg.pinv(R).T
    B = A@A.T
    Theta = np.linalg.inv(B)
    
    P = matrix(np.kron(B,np.eye(d)))
    q = matrix(np.zeros(d*(k-1)))
    G = matrix(-XcheckT)
    h = matrix(-np.ones(XcheckT.shape[0]))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h)
    u = np.array(sol['x'])
    U = u.reshape(d,n_classes-1, order='F')
    return U



def FnormA_du_perm(X,y,n_classes):
    k = n_classes
    d = X.shape[1]
    n = len(y)
    K = linear_kernel(X)
    
    P = matrix(dual_quad_matrix(K, y, k))
    q = matrix(-np.ones(((k-1)*n,)))
    G = matrix(-np.eye(len(q)))
    h = matrix(np.zeros(len(q)))
        
    
    sol = solvers.qp(P,q,G,h)
    beta = np.array(sol['x'])
    alpha = beta.reshape(-1,n_classes-1)
    
    
    am = get_AMPLE_matrices(k)

    ones_column = np.ones((k-1, 1))
    ident = np.eye(k-1)
    R = np.hstack((ones_column, -ident))
    A = np.linalg.pinv(R).T
    B = A@A.T
    Theta = np.linalg.inv(B)

    Uhat_list = [X[i,:].reshape(-1,1)@alpha[i,:].reshape(1,-1)@am[y[i]]@Theta for i in range(n)]
    
    Uhat = np.sum(Uhat_list,axis=0)
    
    What = Uhat@A
    
    return (What, alpha)





def FnormA_du_orig(X,y,n_classes):
    k = n_classes
    d = X.shape[1]
    n = len(y)

    K = linear_kernel(X)   
    
    keep = np.mod(np.arange(0,n*k),k)!= np.repeat(y,k)
    P = matrix(np.kron(K, np.eye(k)))

    q = matrix((np.eye(k)[y,:] - np.ones((n,k))).flatten())


    G = matrix(-np.eye(len(q))[keep,:])
    h = matrix(np.zeros(n*(k-1)))

    A = matrix(np.kron( np.eye(n), np.ones((1,k))))
    b = matrix(np.zeros(len(y)))
    
    
    sol = solvers.qp(P,q,G,h,A,b)
    beta = np.array(sol['x'])
    alpha = beta.reshape(-1,k)
    

    What_list = [(X[i,:].reshape(-1,1))@(alpha[i,:].reshape(1,-1)) for i in range(n)]


    What = -np.sum(What_list,axis=0)

    
    return (What, alpha)