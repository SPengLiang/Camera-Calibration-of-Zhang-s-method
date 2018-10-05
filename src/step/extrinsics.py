#获取每一幅图的外参矩阵[R|t]

import numpy as np

def get_extrinsics_parm(H, intrinsics_parm):
    extrinsics_parm = []

    inv_intrinsics_parm = np.linalg.inv(intrinsics_parm)
    for i in range(len(H)):
        h0 = (H[i].reshape(3, 3))[:, 0]
        h1 = (H[i].reshape(3, 3))[:, 1]
        h2 = (H[i].reshape(3, 3))[:, 2]

        lampda = 1 / np.linalg.norm(np.dot(inv_intrinsics_parm, h0))
        r0 = lampda * np.dot(inv_intrinsics_parm, h0)
        r1 = lampda * np.dot(inv_intrinsics_parm, h1)
        t = lampda * np.dot(inv_intrinsics_parm, h2)
        r2 = np.cross(r0, r1)

        R = np.array([r0, r1, r2, t]).transpose()
        extrinsics_parm.append(R)

    return extrinsics_parm