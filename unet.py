from keras.models import *
from keras.layers import *
from keras.layers.core import Activation


def unet(keep_prob,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv1 64*128*128
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv2 128*64*64
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 第三个卷积层，输出尺度[1, 32, 32, 256]
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 第四个卷积层，输出尺度[1, 16, 16, 512]
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Activation('relu')(conv5)
    # 第五个卷积层，输出尺度[1, 8, 8, 1024]
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    # 第一个反卷积，输出尺度 [1,16,16,512]
    merge6 = concatenate([conv4, up6], axis=3)
    # 第一个反卷积连接第四层[1,16,16,512]+[1,16,16,512]=[1,16,16,1024]
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    # merge6 [1,16,16,1024]
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)

    conv6 = Activation('relu')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    # 第二个反卷积，输出尺度 [1,32,32,256]
    merge7 = concatenate([conv3, up7], axis=3)
    # 第二个反卷积连接第三层[1, 32, 32, 256]+[1, 32, 32, 256]=[1, 32, 32, 512]
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # merge7 [1, 32, 32, 512]
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    # 第三个反卷积，输出尺度 [1,64,64,128]
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    # 第三个反卷积，输出尺度 [1,64,64,128]+ [1,64,64,128]
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model
