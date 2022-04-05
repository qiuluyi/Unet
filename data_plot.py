import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def excel_one_line_to_list():
    df = pd.read_excel('unet_result.xls', names=None,usecols=np.arange(1, 4, 1),)  # 读取项目名称列,不要列名,usecols表示读取的哪几列
    rows = df.shape[0]
    df_li = df.values.tolist()
    unet_loss = []
    unet_jarccard = []
    unet_dice = []
    for s_li in df_li:
        unet_loss.append(s_li[2])
        unet_jarccard.append(s_li[0])
        unet_dice.append(s_li[0])
    return rows, unet_loss,unet_jarccard, unet_dice


if __name__ == '__main__':
    rows, unet_loss,unet_jarccard,unet_dice=excel_one_line_to_list()
    plt.figure(num=1)
    x = np.linspace(1, rows, rows)
    l1, = plt.plot(x, unet_loss, color="red", linewidth=1.5, linestyle="-")
    plt.legend(handles=[l1],labels=['unet_loss','runet_loss','ours1_loss','ours2_loss'],loc="best")
    plt.title("test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("unet_loss.png")

    plt.figure(num=2)
    x = np.linspace(1, rows, rows)
    l1, = plt.plot(x, unet_jarccard, color="red", linewidth=1.5, linestyle="-")
    plt.legend(handles=[l1],labels=['unet_jarccard','runet_jarccard','ours1_jarccard','ours2_jarccard'],loc="best")
    plt.title("jarccard")
    plt.xlabel("epoch")
    plt.ylabel("jarccard")
    plt.savefig("unet_jarccard.png")

    plt.figure(num=3)
    x = np.linspace(1, rows, rows)
    l1, = plt.plot(x, unet_dice, color="red", linewidth=1.5, linestyle="-")
    plt.legend(handles=[l1], labels=['unet_dice', 'runet_dice', 'ours1_dice', 'ours2_dice'],
               loc="best")
    plt.title("dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.savefig("unet_dice.png")