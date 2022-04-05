
"""将图片转为npy格式"""

import cv2 as cv
import numpy as np
import glob
import os


def make_filename_list(file, input_dir):
    img = []
    label = []
    for sub_dir in file:
            # 获取一个子目录中所有的图片文件
            extensions = ['jpg', 'jpeg', 'png']     # 列出所有扩展名，windows不区分大小写，linux区分
            file_list = []
            # os.path.basename()返回path最后的文件名。若path以/或\结尾，那么就会返回空值
            dir_name = os.path.basename(sub_dir)    # 返回子文件夹的名称（sub_dir是包含文件夹地址的串，去掉其地址，只保留文件夹名称）
            if dir_name == input_dir:
                continue
            # 针对不同的扩展名，将其文件名加入文件列表
            for extension in extensions:
                # input_dir是数据集的根文件夹，其下有两个子文件夹，分别是图片和label；
                # dir_name是这次循环中存放所要处理的某种花的图片的文件夹的名称
                # file_glob形如"INPUT_DATA/dir_name/*.extension"
                file_glob = os.path.join(input_dir, dir_name, "*." + extension)
                # extend()的作用是将glob.glob(file_glob)加入file_list
                # glob.glob()返回所有匹配的文件路径列表,此处返回的是所有在input_dir/dir_name文件夹中，且扩展名是extension的文件
                file_list.extend(glob.glob(file_glob))
                file_list.sort()
            # 猜想这句话的意思是，如果file_list是空list，则不继续运行下面的数据处理部分，而是直接进行下一轮循环，
            # 即换一个子文件夹继续操作
            if not file_list:
                continue
            print(file_list)
            print("文件名列表制作完毕，开始读取图片文件")

            for file_name in file_list:
                # 以下两行是读文件常用语句
                if "img" in dir_name:
                    img.append(file_name)
                else:
                    label.append(file_name)
            print("本类图片读取完毕")
    return img, label


def make_npy_list(file_list):
    list = []
    for file in file_list:
        print("正在制作npy文件....", file)
        image = cv.imread(file, flags=0)

        image = np.reshape(image, [image.shape[0], image.shape[1], 1])
        list.append(image)
    return list


def make_train_npy():
    sub_img_dirs = [x[0] for x in os.walk("./train_images")]  # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    train_img, train_label = make_filename_list(sub_img_dirs, "train_images")
    if os.path.exists('./npy_data'):
        pass
    else:
        os.makedirs("./npy_data")
    train_img_list = make_npy_list(train_img)
    np.save("npy_data/train_images", train_img_list)
    print("训练图片的npy文件制作完成")
    train_label_list = make_npy_list(train_label)
    np.save("npy_data/train_masks", train_label_list)
    print("训练label的npy文件制作完成")


def make_test_npy():
    sub_img_dirs = [x[0] for x in os.walk("./test_images")]  # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    train_img, train_label = make_filename_list(sub_img_dirs, "test_images")
    if os.path.exists('./npy_data'):
        pass
    else:
        os.makedirs("./npy_data")
    train_img_list = make_npy_list(train_img)
    np.save("npy_data/test_images", train_img_list)
    print("测试图片的npy文件制作完成")
    train_label_list = make_npy_list(train_label)
    np.save("npy_data/test_masks", train_label_list)
    print("测试label的npy文件制作完成")


if __name__ == "__main__":
    make_train_npy()
    make_test_npy()