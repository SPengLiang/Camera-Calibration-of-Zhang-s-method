# Camera-Calibration-of-Zhang-s-method
Zhang‘s method Camera Calibration Implement by numpy


   利用张正友法进行相机参数标定，所用语言为python，所需依赖包有numpy，scipy， OpenCV，运行前需要事先安装好相应工具包。标定相机所用的图像放在pic文件夹下，下载到本地之后，运行main.py文件即可
   
   标定步骤主要分为求单应矩阵，求相机内参，求对应每幅图外参，求畸变矫正系数，微调所有参数等五个步骤，每个步骤对应一个py文件，放置于src\step文件夹下。
