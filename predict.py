import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import serializers
import chainer.links as L

from network import NN

if __name__ == '__main__':
    # モデルの読み込み
    model = L.Classifier(NN(100, 10))
    serializers.load_npz('digit_recognizer.model', model)

    # 推定対象の読み込み
    test_data = np.genfromtxt('data/test.csv',
                              delimiter=',', skip_header=1) # data.shape => (28000, 784)
    target = test_data[np.random.randint(0, 28000)].astype(np.float32)

    # 推定対象の確認（画像表示）
    plt.imshow(np.reshape(target, (28, 28)), cmap='gray') # 対象の画像を表示
    plt.show()

    # 推定
    prediction = model.predictor(chainer.Variable(np.array([target]))).data[0]
    print(prediction.argmax())
