from .layer import Layer
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


class Classifier:
    def __init__(self, loss, epochs, optimizer, batch_size):
        self.input_layer = None
        self.layers = []
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.batch_size = batch_size
        self.error = 0.0
        self.predictions = np.array([])
        self.outputs = np.array([])
        self.val_predictions = np.array([])
        self.val_outputs = np.array([])

    def add_layer(self, layer):
        if layer.input_dim is not None:
            self.input_layer = Layer(neurons_num=layer.input_dim, activation='')
            self.layers.append(self.input_layer)
        else:
            layer.add_connection(self.layers[-1])
            self.layers.append(layer)

    def fit(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        for i in range(self.epochs):
            self.do_epoch()
            self.test()
            acc, loss, val_acc, val_loss = self.get_accuracies()
            print('epoch %d/%d -' % (i + 1, self.epochs), 'loss: %.4f -' % loss,
                  'accuracy: %.4f -' % acc, 'val_loss: %.4f -' % val_loss,
                  'val_accuracy: %.4f' % val_acc)

    def do_epoch(self):
        for n in range(0, len(self.X_train), self.batch_size):
            if n + self.batch_size <= len(self.X_train):
                batch = self.X_train[n:n + self.batch_size, :]
                vals = self.y_train[n:n + self.batch_size]
            else:
                break
            self.feed_forward(batch)
            self.get_predictions(vals)
            self.back_propagate()

    def test(self):
        for n in range(0, len(self.X_test), self.batch_size):
            if n + self.batch_size <= len(self.X_test):
                batch = self.X_test[n:n + self.batch_size, :]
                vals = self.y_test[n:n + self.batch_size]
            else:
                break
            self.feed_forward(batch)
            self.get_predictions(vals, is_test=True)

    def feed_forward(self, row):
        for v, n in zip(row.T, self.input_layer.neurons):
            n.output = v

        for l in self.layers[1:]:
            for n in l.neurons:
                n.feed_forward()

    def get_predictions(self, val, is_test=False):
        if len(self.layers[-1].neurons) > 1:
            for n, y in zip(self.layers[-1].neurons, val):
                n.get_error(self.loss, val)
        else:
            self.layers[-1].neurons[0].get_error(self.loss, val)

        for n in self.layers[-1].neurons:
            pred = n.output > 0.5
            pred.astype(np.int)
            if is_test:
                self.val_outputs = np.concatenate((self.val_outputs, n.output))
                self.val_predictions = np.concatenate((self.val_predictions, pred))
            else:
                self.outputs = np.concatenate((self.outputs, n.output))
                self.predictions = np.concatenate((self.predictions, pred))

    def back_propagate(self):
        for l in self.layers[::-1]:
            for n in l.neurons:
                n.back_propagate(optimizer=self.optimizer)

    def predict(self, x):
        self.feed_forward(x)
        return self.predictions

    def get_accuracies(self):
        acc = accuracy_score(self.y_train[0:len(self.predictions)], self.predictions)
        val_acc = accuracy_score(self.y_test[0:len(self.val_predictions)], self.val_predictions)
        loss = mean_squared_error(self.y_train[0:len(self.outputs)], self.outputs)
        val_loss = mean_squared_error(self.y_test[0:len(self.val_outputs)], self.val_outputs)
        self.predictions = np.array([])
        self.val_predictions = np.array([])
        self.outputs = np.array([])
        self.val_outputs = np.array([])
        return acc, loss, val_acc, val_loss
