import numpy as np

def get_distortion(intrinsic_pram, extrinsic_pram, pic_coor, real_coor):
    D = []
    d = []
    for i in range(len(pic_coor)):
        for j in range(len(pic_coor[i])):
            single_coor = np.array([(real_coor[i])[j, 0], (real_coor[i])[j, 1], 0, 1])
            u = np.dot(np.dot(intrinsic_pram, extrinsic_pram[i]), single_coor)
            [u_estm, v_estm] = [u[0]/u[2], u[1]/u[2]]

            r = np.linalg.norm((real_coor[i])[j])
            D.append(np.array([(u_estm - intrinsic_pram[0, 2]) * r ** 2, (u_estm - intrinsic_pram[0, 2]) * r ** 4]))
            D.append(np.array([(v_estm - intrinsic_pram[1, 2]) * r ** 2, (v_estm - intrinsic_pram[1, 2]) * r ** 4]))

            d.append(pic_coor[i][j, 0] - u_estm)
            d.append(pic_coor[i][j, 1] - v_estm)

    D = np.array(D)

    U, S, Vh=np.linalg.svd(D, full_matrices=0)

    temp_S = np.array([[S[0], 0],
                       [0, S[1]]])
    temp_res = np.dot(Vh.transpose(), np.linalg.inv(temp_S))
    temp_res_res = np.dot(temp_res, U.transpose())
    k = np.dot(temp_res_res, d)

    return k