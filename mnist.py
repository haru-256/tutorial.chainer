import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import (Chain, ChainList, Function, Variable, datasets,
                     gradient_check, iterators, link, optimizers, report,
                     serializers, training, utils)
from chainer.training import extensions


# define model
class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of input to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(x))
        logits = self.l3(h2)
        return logits

class Classifier(Chain):
    def __init__(self, predictor):
        """
        predictor: chain
           this is model
        """
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
            
    def __call__(self, x, t):
        """
        x, t: variable
           these are input data
        """
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)  # define loss function
        accuracy = F.accuracy(y, t)  # calculate metrics
        report({'loss': loss, 'accuracy': accuracy}, self)  # report loss and accuracy to trainer
        return loss

# obtain mnist datasets
train, test = datasets.get_mnist()
train, test

# make iterator
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(
    test, batch_size=100, shuffle=False, repeat=False)

model = L.Classifier(MLP(100, 10))  # the input size, 784, is infferd
optimizer = optimizers.SGD()  # define optimizer
optimizer.setup(model)
# optimizer.add_hook(chainer.optimizer.WeightDecay(0.00005))

updater = training.StandardUpdater(train_iter, optimizer)  # forward/backward coumputation & Parametor update
trainer = training.Trainer(updater, (20, "epoch"), out="result")  # whole procedure

# evaluate model at end of every epoch
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())  # output repoted value
trainer.extend(extensions.PrintReport(
    ["epoch", "main/accuracy", "validation/main/accuracy"]))  # Print logreport
trainer.extend(extensions.ProgressBar())  # show the progress bar

trainer.run()  # 20 epoch だけ訓練差させる
