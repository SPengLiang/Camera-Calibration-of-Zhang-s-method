import cv2 as cv
import numpy as np
import os
from step.homography import get_homography
from step.intrinsics import get_intrinsics_parm
from step.extrinsics import get_extrinsics_parm
from step.distortion import get_distortion
from step.refine_all import refinall_all_prams


def calibrate():
    H = get_homography(pic_points, real_points_x_y)
    intrinsics_parm = get_intrinsics_parm(H)
    extrinsics_parm = get_extrinsics_parm(H, intrinsics_parm)

    k = get_distortion(intrinsics_parm, extrinsics_parm, pic_points, real_points_x_y)

    [new_intrinsics_parm, new_k, new_extrinsics_parm]  = refinall_all_prams(intrinsics_parm,
                                                            k, extrinsics_parm, real_points, pic_points)

    print("intrinsics_parm:\t", new_intrinsics_parm)
    print("new_k:\t", new_k)
    print("extrinsics_parm:\t", new_extrinsics_parm)


if __name__ == "__main__":
    file_dir = r'D:\stereo\stereo\Project_Stereo_left\left'
    # 标定所用图像
    pic_name = os.listdir(file_dir)

    # 由于棋盘为二维平面，设定世界坐标系在棋盘上，一个单位代表一个棋盘宽度，产生世界坐标系三维坐标
    real_coor = np.zeros((9 * 6, 3), np.float32)
    real_coor[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    real_points = []
    real_points_x_y = []
    pic_points = []

    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path)

        # 寻找到棋盘角点
        succ, pic_coor = cv.findChessboardCorners(pic_data, (9, 6), None)

        if succ:
            # 添加每幅图的对应3D-2D坐标
            pic_coor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)

            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])
    calibrate()
