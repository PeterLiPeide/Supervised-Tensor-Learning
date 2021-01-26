"""Tensor Discriminant Analysis function
Li, Q., & Schonfeld, D. (2014). Multilinear discriminant analysis for higher-order tensor data classification. IEEE transactions on pattern analysis and machine intelligence, 36(12), 2524-2537.

Include both DGTDA and CMDA as tensor discriminant analysis classifier

Dependiencies: numpy 
               tensorly
"""

import numpy as np 
import tensorly as tl 


def DGTDA(X, y, **kwargs):
    """
    Direct General Tensor Discriminant Analysis
    Details see reference

    Input: X: list of tensor predictors 
           y: list of labels (-1, 1) or (0, 1)
    Optional: p_dim : a list of projected dimension; default is int(0.7 * I_d)

    Output: a list of project matrices 

    Notice the returned projection matrices are in the shape of [P_d, I_d]. This is consistent 
    to the tensor mode product defined in the Kolda's tensor decomposition and application paper 
    and it slightly different from the notation in the DGTDA paper. U is the left singular vector, where the dimension should be I_d. 
    """

    ts_size = list(X[0].shape)
    if "p_dim" in kwargs:
        p_dim = kwargs['p_dim']
    else:
        p_dim = [int(0.7 * i) for i in ts_size]
    U_list = []

    positive_group = np.array([X[i] for i in y if i > 0])
    negative_group = np.array([X[i] for i in y if i <= 0])
    positive_mean = np.mean(positive_group, axis=0)
    negative_mean = np.mean(negative_group, axis=0)
    n1 = positive_group.shape[0]
    n2 = negative_group.shape[0]
    global_mean = (positive_mean * n1 + negative_mean * n2) / len(X)

    for d in range(len(ts_size)):
        B1 = tl.unfold(positive_mean - global_mean, d)
        B2 = tl.unfold(negative_mean - global_mean, d)
        B = n1 * np.dot(B1, B1.T) + n2 * np.dot(B2, B2.T)

        W = 0
        for j in range(len(X)):
            if y[j] > 0:
                tmp = tl.unfold(X[j] - positive_mean, d)
            else:
                tmp = tl.unfold(X[j] - negative_mean, d) 
            W += np.dot(tmp, tmp.T)
        _, S, _ = np.linalg.svd(np.dot(np.linalg.pinv(W), B))
        # get the maximum singular value
        eta = S[0]   
        Td = B - eta * W
        U, _, _ = np.linalg.svd(Td)
        # First P many columns as projection
        U_list.append(U[:, :p_dim[d]].T)  
    
    return U_list


def CMDA(X, y, **kwargs):
    """
    Constrained Multilinear Discriminant Analysis
    see details in the reference

    Input: X: list of tensor predictors 
           y: list of labels (-1, 1) or (0, 1)
    Optional: p_dim : a list of projected dimension; default is int(0.7 * I_d)
              Maxitor: maximum iteration time
              eta: stopping creatia
              U_list: Initial value of U


    Output: a list of project matrices 

    Notice the returned projection matrices are in the shape of [P_d, I_d]. This is consistent to the tensor mode 
    product defined in the Kolda's tensor decomposition and application paper 
    and it slightly different from the notation in the DGTDA paper. U is the left singular vector, where the dimension should be I_d. 
    """
    ts_size = list(np.shape(X[0]))

    if "p_dim" in kwargs:
        p_dim = kwargs['p_dim']
    else:
        p_dim = [int(0.7 * i) for i in ts_size]
    if 'Max_itor' in kwargs:
        Max_itor = kwargs['Max_itor']
    else:
        Max_itor = 10
    if 'U_list' in kwargs:
        U_list = kwargs['U_list']
    else:
        U_list = [np.ones((i, j)) for i, j in zip(p_dim, ts_size)]
    if "eta" in kwargs:
        eta = kwargs['eta']
    else:
        eta = 0.01
    
    iterator = 0
    current_eta = 1000
    positive_group = np.array([X[i] for i in y if i > 0])
    negative_group = np.array([X[i] for i in y if i <= 0])
    positive_mean = np.mean(positive_group, axis=0)
    negative_mean = np.mean(negative_group, axis=0)
    n1 = positive_group.shape[0]
    n2 = negative_group.shape[0]
    global_mean = (positive_mean * n1 + negative_mean * n2) / len(X)

    while iterator < Max_itor and current_eta > eta:
        U_current = U_list[:]
        for d in range(len(ts_size)):
            B1 = tl.tenalg.multi_mode_dot(positive_mean - global_mean, U_current, skip=d)
            B1 = tl.unfold(B1, d)
            B2 = tl.tenalg.multi_mode_dot(negative_mean - global_mean, U_current, skip=d)
            B2 = tl.unfold(B2, d)
            B = n1 * np.dot(B1, B1.T) + n2 * np.dot(B2, B2.T)

            W = 0
            for j in range(len(X)):
                if y[j] > 0:
                    tmp = tl.tenalg.multi_mode_dot(X[j] - positive_mean, U_current, skip=d)
                else:
                    tmp = tl.tenalg.multi_mode_dot(X[j] - negative_mean, U_current, skip=d)  
                tmp = tl.unfold(tmp, d)  
                W += np.dot(tmp, tmp.T)
            Td = np.dot(np.linalg.inv(W), B)
            Ud, _, _ = np.linalg.svd(Td)
            U_current[d] = Ud[:, : p_dim[d]].T
        current_eta = sum([np.linalg.norm(np.dot(a, b.T) - np.eye(np.shape(a)[0])) for a, b in zip(U_current, U_list)])
        U_list = U_current
        iterator += 1
    return U_list


"""
Projecting tensor can be easily done by
Xnew = tl.tenalg.mult_mode_dot(X, U_list) 

Thus, we only provide a KNN classifier here
"""


def Tensor_NN(X, y, X_test, K):
    """
    K nearest neighbour classifier for tensor data
    """
    y_test = []
    for val in X_test:
        tmp_dis = []
        for Xa in X:
            tmp_dis.append(np.linalg.norm(val - Xa))
        idx = np.argsort(tmp_dis).tolist()
        idx = idx[: K]
        neighbours = [y[i] for i in idx]
        y_pred = 1 if np.mean(neighbours) > 0 else 0
        y_test.append(y_pred)
    return y_test
