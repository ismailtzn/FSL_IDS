import logging

import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout
from keras.models import Sequential


class AttackClassifier:

    def __init__(self, exp_params):
        self.input_nodes = 78
        self.output_nodes = exp_params['output_nodes']
        self.layer_nodes = [64]
        self.activations = ['relu']
        self.dropouts = [0.2]

        self.batch_size = exp_params['batch_size']
        self.epochs = exp_params['epochs']

        self.ann = None
        self.create_ann()

    def reinit(self, new_output_node_count):
        self.output_nodes = new_output_node_count
        # self.create_ann()

        self.ann.pop()
        self.ann.add(Dense(self.output_nodes, activation='softmax',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def create_ann(self):
        self.ann = Sequential()
        # First hidden layer (need to specify input layer nodes here)
        self.ann.add(Dense(self.layer_nodes[0], input_dim=self.input_nodes, activation=self.activations[0],
                           kernel_initializer='he_uniform', bias_initializer='zeros'))
        self.ann.add(BatchNormalization())
        self.ann.add(Dropout(self.dropouts[0]))

        # Other hidden layers
        for i in range(1, len(self.layer_nodes)):
            self.ann.add(Dense(self.layer_nodes[i], activation=self.activations[i],
                               kernel_initializer='he_uniform', bias_initializer='zeros'))
            self.ann.add(BatchNormalization())  # BN after Activation (contested, but current preference)
            self.ann.add(Dropout(self.dropouts[i]))

        # Output layer
        self.ann.add(Dense(self.output_nodes, activation='softmax',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        logger = logging.getLogger(__name__)
        self.ann.summary(print_fn=logger.info)

    def fit(self, X_train, y_train):
        # # --------------------------------
        # # Class weights to according to imbalance
        # logging.info("Computing class weights to according to imbalance")
        # y_ints = [y.argmax() for y in y_train]
        # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_ints), y=y_ints)
        # d_class_weights = dict(enumerate(class_weights))
        # logging.info("Class weights below.\n{}".format(class_weights))
        # # --------------------------------
        # # Fit
        # history = self.ann.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
        #                        class_weight=d_class_weights, verbose=1)
        history = self.ann.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return history

    def predict(self, X):
        return self.ann.predict(X, verbose=0)

    def predict_classes(self, X):
        predict_x = self.ann.predict(X, verbose=0)
        classes_x = np.argmax(predict_x, axis=1)
        return classes_x

    def save(self, filename, **kwargs):
        self.ann.save(filename, kwargs)
