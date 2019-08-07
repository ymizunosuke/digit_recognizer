from chainer import Chain
import chainer.functions as F
import chainer.links as L


class NN(Chain):
    def __init__(self, n_units, n_out):
        super(NN, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
