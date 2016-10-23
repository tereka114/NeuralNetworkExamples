#coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import chainer
import chainer.datasets
from chainer import training
from chainer.training import extensions

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

if __name__ == '__main__':
    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, 32)
    test_iter = chainer.iterators.SerialIterator(test, 32,
                                                 repeat=False, shuffle=False)
    model = L.Classifier(MLP(784, 10))
    optimizer = chainer.optimizers.Adam()

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (10, 'epoch'), out="result")

    trainer.extend(extensions.Evaluator(test_iter, model, device=10))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    trainer.extend(extensions.ProgressBar())
    trainer.run()