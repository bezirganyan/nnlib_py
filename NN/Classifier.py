from .layer import Layer


class Classifier:
    def __init__(self, loss, batch_size, epochs):
        # TODO - optimizer
        self.input_layer = None
        self.layers = []
        self.loss = loss
        # self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.error = 0.0

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

        for i in range(10):
            ind = 0
            for row, val in zip(self.X_train, self.y_train):
                self.do_step(row, val)
                # break
                # print(self.error)
                # ind += 1
                # print(ind/len(self.X_train))
            print('epoch', i+1, 'loss', self.error)

    def do_step(self, row, val):
        for v, n in zip(row, self.input_layer.neurons):
            n.output = v

        for l in self.layers[1:]:
            for n in l.neurons:
                n.feed_forward()

        if len(self.layers[-1].neurons) > 1:
            for n, y in zip(self.layers[-1].neurons, val):
                n.get_error(self.loss, val)
        else:
            self.layers[-1].neurons[0].get_error(self.loss, val)

        for n in self.layers[-1].neurons:
            self.error = 0
            self.error += n.error

        for l in self.layers[::-1]:
            for n in l.neurons:
                n.back_propagate()

