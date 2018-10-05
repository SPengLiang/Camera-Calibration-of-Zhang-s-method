import numpy as np
import math
from scipy import optimize as opt

def refinall_all_prams(A, k, W, real_coor, pic_coor):
    P_init = compose_paramter_vector(A, k, W)

    X_double = np.zeros((2*len(real_coor)*len(real_coor[0]), 3))
    Y = np.zeros((2*len(real_coor)*len(real_coor[0])))

    M = len(real_coor)
    N = len(real_coor[0])
    for i in range(M):
        for j in range(N):
            X_double[(i * N + j) * 2] = (real_coor[i])[j]
            X_double[(i * N + j) * 2 + 1] = (real_coor[i])[j]
            Y[(i * N + j) * 2] = (pic_coor[i])[j, 0]
            Y[(i * N + j) * 2 + 1] = (pic_coor[i])[j, 1]

    P = opt.leastsq(value,
                   P_init,
                   args=(real_coor, pic_coor),
                   Dfun=jacobian)[0]

    error = value(P, real_coor, pic_coor)
    raial_error = [np.sqrt(error[2*i]**2 + error[2*i+1]**2) for i in range(len(error) // 2)]

    print("total avg error:\t", sum(raial_error) / (len(error) // 2))

    return decompose_paramter_vector(P)

def compose_paramter_vector(A, k, W):
    alpha = np.array([A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2], k[0], k[1]])
    P = alpha
    for i in range(len(W)):
        R, t = (W[i])[:, :3], (W[i])[:, 3]
        zrou = to_rodrigues_vector(R)
        w = np.append(zrou, t)
        P = np.append(P, w)
    return P

def decompose_paramter_vector(P):
    [alpha, beta, gamma, uc, vc, k0, k1] = P[0:7]
    A = np.array([[alpha, gamma, uc],
                  [0, beta, vc],
                  [0, 0, 1]])
    k = np.array([k0, k1])
    W = []
    M = (len(P) - 7) // 6

    for i in range(M):
        m = 7 + 6 * i
        zrou = P[m:m+3]
        t = (P[m+3:m+6]).reshape(3, -1)
        R = to_rotation_matrix(zrou)
        w = np.concatenate((R, t), axis=1)
        W.append(w)

    W = np.array(W)
    return A, k, W

def get_single_project_coor(A, W, k, coor):
    single_coor = np.array([coor[0], coor[1], coor[2], 1])

    uv = np.dot(np.dot(A, W), single_coor)
    uv /= uv[-1]

    # 透镜矫正
    u0 = uv[0]
    v0 = uv[1]
    r = np.linalg.norm(coor)
    uc = A[0, 2]
    vc = A[1, 2]

    u = u0 - (u0 - uc) * r ** 2 * k[0] - (u0 - uc) * r ** 4 * k[1]
    v = v0 - (v0 - vc) * r ** 2 * k[0] - (v0 - vc) * r ** 4 * k[1]

    return np.array([u, v])

def value(P, X, Y_real):
    M = (len(P) - 7) // 6
    N = len(X[0])
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
    Y = np.array([])

    for i in range(M):
        m = 7 + 6 * i

        w = P[m:m + 6]
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)

        for j in range(N):
            Y = np.append(Y, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))

    error_Y  =  np.array(Y_real).reshape(-1) - Y

    print(sum(error_Y*error_Y))
    return error_Y

def jacobian(P, X, Y_real):
    M = (len(P) - 7) // 6
    N = len(X[0])
    K = len(P)
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])

    res = np.array([])

    for i in range(M):
        m = 7 + 6 * i

        w = P[m:m + 6]
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)

        for j in range(N):
            res = np.append(res, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))

    J = np.zeros((K, 2*M*N))
    for k in range(K):
        '''
        derv_x = np.gradient(res[:, 0], P[k])
        derv_y = np.gradient(res[:, 1], P[k])
        J[k, [2 * i for i in range(M*N)]] = derv_x
        J[k, [2 * i + 1 for i in range(M*N)]] = derv_y
        '''
        J[k] = np.gradient(res, P[k])

    return J.T

def to_rotation_matrix(zrou):
    theta = np.linalg.norm(zrou)
    zrou_prime = zrou / theta

    W = np.array([[0, -zrou_prime[2], zrou_prime[1]],
                  [zrou_prime[2], 0, -zrou_prime[0]],
                  [-zrou_prime[1], zrou_prime[0], 0]])
    R = np.eye(3, dtype='float') + W * math.sin(theta) + np.dot(W, W) * (1 - math.cos(theta))

    return R

def to_rodrigues_vector(R):
    p = 0.5 * np.array([[R[2, 1] - R[1, 2]],
                        [R[0, 2] - R[2, 0]],
                        [R[1, 0] - R[0, 1]]])
    c = 0.5 * (np.trace(R) - 1)

    if np.linalg.norm(p) == 0:
        if c == 1:
            zrou = np.array([0, 0, 0])
        elif c == -1:
            R_plus = R + np.eye(3, dtype='float')

            norm_array = np.array([np.linalg.norm(R_plus[:, 0]),
                                   np.linalg.norm(R_plus[:, 1]),
                                   np.linalg.norm(R_plus[:, 2])])
            v = R_plus[:, np.where(norm_array == max(norm_array))]
            u = v / np.linalg.norm(v)
            if u[0] < 0 or (u[0] == 0 and u[1] < 0) or (u[0] == u[1] and u[0] == 0 and u[2] < 0):
                u = -u
            zrou = math.pi * u
        else:
            zrou = []
    else:
        u = p / np.linalg.norm(p)
        theata = math.atan2(np.linalg.norm(p), c)
        zrou = theata * u

    return zrou