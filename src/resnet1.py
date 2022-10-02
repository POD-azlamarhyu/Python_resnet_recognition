import argparse as arg
import os
import glob
import random,math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl

TEST_DATA_FILE_NUMPY = "dataset.npy"
TEST_DATA_FILE_PKL="dataset.pkl"

# 残差ブロック(Bottleneckアーキテクチャ)
class Res_Block(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        bneck_channels = out_channels // 4

        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv1 = kl.Conv2D(bneck_channels, kernel_size=1, 
                        strides=1, padding='valid', use_bias=False)

        self.bn2 = kl.BatchNormalization()
        self.av2 = kl.Activation(tf.nn.relu)
        self.conv2 = kl.Conv2D(bneck_channels, kernel_size=3, 
                        strides=1, padding='same', use_bias=False)

        self.bn3 = kl.BatchNormalization()
        self.av3 = kl.Activation(tf.nn.relu)
        self.conv3 = kl.Conv2D(out_channels, kernel_size=1, 
                        strides=1, padding='valid', use_bias=False)

        self.shortcut = self._scblock(in_channels, out_channels)
        self.add = kl.Add()

    # Shortcut Connection
    def _scblock(self, in_channels, out_channels):

        if in_channels != out_channels:
            self.bn_sc1 = kl.BatchNormalization()
            self.conv_sc1 = kl.Conv2D(out_channels, kernel_size=1, 
                        strides=1, padding='same', use_bias=False)
            return self.conv_sc1
        else:
            return lambda x : x

    def call(self, x):   

        out1 = self.conv1(self.av1(self.bn1(x)))
        out2 = self.conv2(self.av2(self.bn2(out1)))
        out3 = self.conv3(self.av3(self.bn3(out2)))
        shortcut = self.shortcut(x)
        out4 = self.add([out3, shortcut])

        return out4

# ResNet50(Pre Activation)
class ResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self._layers = [

            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False, input_shape=input_shape),
            kl.MaxPool2D(pool_size=3, strides=2, padding="same"),
            Res_Block(64, 256),
            [
                Res_Block(256, 256) for _ in range(2)
            ],
            kl.Conv2D(512, kernel_size=1, strides=2),
            [
                Res_Block(512, 512) for _ in range(4)
            ],
            kl.Conv2D(1024, kernel_size=1, strides=2, use_bias=False),
            [
                Res_Block(1024, 1024) for _ in range(6)
            ],
            kl.Conv2D(2048, kernel_size=1, strides=2, use_bias=False),
            [
                Res_Block(2048, 2048) for _ in range(3)
            ],
            kl.GlobalAveragePooling2D(),
            kl.Dense(1000, activation="relu"),
            kl.Dense(output_dim, activation="softmax")
        ]

    def call(self, x):
        for layer in self._layers:

            if isinstance(layer, list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)

        return x

# 学習
class trainer(object):
    def __init__(self):

        self.resnet = ResNet((28, 28, 1), 10)
        self.resnet.build(input_shape=(None, 28, 28, 1))
        self.resnet.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, train_img, train_lab, test_images, test_labels, out_path, batch_size, epochs):

        print("\n\n___Start training...")

        his = self.resnet.fit(train_img, train_lab, batch_size=batch_size, epochs=epochs)

        graph_output(his, out_path) # グラフ出力

        print("___Training finished\n\n")

        self.resnet.evaluate(test_images, test_labels) # テストデータ推論

        print("\n___Saving parameter...")
        out_path = os.path.join(out_path, "resnet.h5")
        self.resnet.save_weights(out_path) # パラメータ保存
        print("___Successfully completed\n\n")

# accuracy, lossグラフ
def graph_output(history, out_path):

    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(out_path, "acc_graph.jpg"))
    plt.show()  

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(out_path, "loss_graph.jpg"))
    plt.show()


def makeDataSet(file):
    global X,Y
    X=[]
    Y=[]

    for cat,fname in file:
        img=Image.open(fname)
        img=img.convert("RGB")
        img=img.resize((224,224))
        data=np.asarray(img)
        X.append(data)
        Y.append(cat)

    return np.array(X),np.array(Y)




def dataSet():
    category=["dryer","kettle","officechair","scissors","outlet"]
    dir="./img/"
    allfiles=[]

    for idx,cat in enumerate(category):
        imgdir=dir+cat
        files=glob.glob(imgdir+"/*.png")

        print("image number : {} {}".format(cat,len(files)))

        for f in files:
            allfiles.append((idx,f))

    random.shuffle(allfiles)
    th=math.floor(len(allfiles)*0.8)
    train=allfiles[0:th]
    test=allfiles[th:]
    X_train,Y_train=makeDataSet(train)
    X_test,Y_test=makeDataSet(test)

    dataFile=(X_train,X_test,Y_train,Y_test)

    np.save(dir+TEST_DATA_FILE_NUMPY,dataFile)
    with open(dir+TEST_DATA_FILE_PKL,"wb") as f:
        pickle.dump(dataFile,f)

def main():

    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='ResNet50')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='パラメータの保存先指定(デフォルト値=./resnet.h5')
    parser.add_argument('--batch_size', '-b', type=int, default=256,
                        help='ミニバッチサイズの指定(デフォルト値=256)')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='学習回数の指定(デフォルト値=40)')
    args = parser.parse_args()

    # 設定情報出力
    print("=== Setting information ===")
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Epoch: {}".format(args.epoch))
    print("===========================")

    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)

    # Fashion-MNIST読込
    if os.path.exists(TEST_DATA_FILE_NUMPY) == False:
        dataSet()
    else:
        pass

    dir="./img/"
    
    (train_images, train_labels), (test_images, test_labels) = np.load(dir+TEST_DATA_FILE_NUMPY,allow_pickle=True)

    # 画像データ加工
    train_imgs = train_images / 255.0
    train_imgs = train_imgs[:, :, :, np.newaxis]
    test_imgs = test_images / 255.0
    test_imgs = test_imgs[:, :, :, np.newaxis]

    Trainer = trainer()
    Trainer.train(train_imgs, train_labels, test_imgs, test_labels, args.out, args.batch_size, args.epoch)

if __name__ == "__main__":
    main()