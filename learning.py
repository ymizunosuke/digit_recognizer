"""KaggleのDigit RecognizerをChainerでやってみた"""

import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import datasets
from chainer import iterators
from chainer import training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F

from network import NN

if __name__ == '__main__':

    # データ整形
    data = np.genfromtxt('data/train.csv',
                         delimiter=',', skip_header=1) # data.shape => (42001, 785)
    x = data[:, 1:].astype(np.float32)
    y = data[:, 0].astype(np.int32)[:, None]
    y = np.ndarray.flatten(y)

    train, test = datasets.split_dataset_random(datasets.TupleDataset(x, y), int(x.shape[0] * .7))

    # iterator 作成
    train_itr = iterators.SerialIterator(train, 100)
    test_itr = iterators.SerialIterator(
        test, 100, repeat=False, shuffle=False)

    # ｍodel 作成
    model = L.Classifier(
        NN(100, 10), lossfun=F.softmax_cross_entropy)
    # どこかのソースから拾ってきて sigmoid_cross_entropy, binary_accuracy を設定したらハマった
    # しっかりドキュメント読むこと

    # optimizer 作成
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # updater 作成
    updater = training.updaters.StandardUpdater(train_itr,
                                                optimizer, device=-1)

    # trainer 作成
    trainer = training.Trainer(updater, (20, 'epoch'), out='results')
    trainer.extend(extensions.Evaluator(test_itr, model, device=-1)) # testデータセットで評価
    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', filename='loss.png')
        )
        trainer.extend(
            extensions.PlotReport(['main/accuracy', 'validation/main/accuraccy'],
                                  'epoch', filename='accuracy.png')
        )

    # 学習実行
    trainer.run()

    # 学習を行ったmodelの保存
    chainer.serializers.save_npz('digit_recognizer.model', model)


