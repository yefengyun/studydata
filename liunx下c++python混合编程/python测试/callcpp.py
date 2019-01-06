import cv2
import os
import numpy as np

os.path.realpath('.')
import libImageProcessAPI as llp
# import ImageProcessAPI as llp


if __name__ == '__main__':
    ip = llp.ImageProcess()  # 获取对象
    # 测试是否加载py库成功
    print(ip.test())

    ip.img_path="./test7.jpg"
    print(ip.img_path)

    img=cv2.imread("./test7.jpg")

    dst=ip.ImageCorrect(img)
    print(type(dst))
    cv2.imshow("show",dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
