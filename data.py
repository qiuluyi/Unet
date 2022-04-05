from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import os
import make_data


def load_train_data():
    print('-' * 30)
    print('load train images...')
    print('-' * 30)
    imgs_train = np.load("npy_data/train_images.npy")
    mask_train = np.load("npy_data/train_masks.npy")
    imgs_train = imgs_train.astype('float32')/255
    mask_train = mask_train.astype('float32')/255
    # imgs_train /= 255
    # mean = imgs_train.mean(axis = 0)
    # imgs_train -= mean
    # imgs_vessels_train /= 255
    # imgs_vessels_train[imgs_vessels_train > 0.5] = 1
    # imgs_vessels_train[imgs_vessels_train <= 0.5] = 0
    return imgs_train, mask_train


def load_val_data():
    print('-' * 30)
    print('load test images...')
    print('-' * 30)
    imgs_val = np.load("npy_data/test_images.npy")
    # imgs_vessels_test = np.load("npydata/imgs_vessels_test.npy")
    mask_val = np.load("npy_data/test_masks.npy")
    imgs_val = imgs_val.astype('float32')/255
    mask_val = mask_val.astype('float32')/255
    # imgs_mask_test = imgs_mask_test.astype('float16')
    # imgs_test /= 255
    # imgs_vessels_test /= 255
    # imgs_vessels_test[imgs_vessels_test > 0.5] = 1
    # imgs_vessels_test[imgs_vessels_test <= 0.5] = 0
    # imgs_mask_test[imgs_mask_test > 0.5] = 1
    # imgs_mask_test[imgs_mask_test <= 0.5] = 0
    # mean = imgs_test.mean(axis = 0)
    # imgs_test -= mean
    return imgs_val, mask_val


if __name__ == "__main__":
    if os.path.exists("npy_data/train_images.npy") and os.path.exists("npy_data/train_masks.npy"):
        pass
    else:
        make_data.make_train_npy()
    if os.path.exists("npy_data/test_images.npy") and os.path.exists("npy_data/test_masks.npy"):
        pass
    else:
        make_data.make_test_npy()
    load_train_data()
    load_val_data()