import keras.layers
from keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np


def train_GRU_1DConv(x_train, y_train, x_test, y_test, scaler, seq_size):
    """Trains a model with two LSTM layers and one dense output layer

    :param x_train: training data
    :param y_train: goal data for training data
    :param x_test: test data
    :param y_test: goal data for test data
    :param scaler: the scaler to inverse the scaling
    :param seq_size: how many timesteps are known
    """

    model = Sequential([
        keras.layers.GRU(20, input_shape=[None, seq_size], return_sequences=True),
        keras.layers.GRU(20),
        keras.layers.Dense(1, activation='tanh')
    ])
    # return_sequence = True, if you use second layer
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # summary gives the shape and parameters of the model
    # model.load_weights('Weights/GRU_1DConv_weights')
    model.summary()
    # history to visualise the loss, epochs the number of fits for the model
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
    y_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    model.save_weights('Weights/GRU_1DConv_weights')
    # model.score(x_result, y_result)
    y_predict = scaler.inverse_transform(y_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # plot baseline and predictions
    plt.plot(np.array(range(0, len(y_predict))), y_predict, color='r')
    plt.plot(np.array(range(0, len(y_train))), y_train, color='b')
    plt.show()
    plt.plot(np.array(range(0, len(test_predict))), test_predict, color='g')
    plt.plot(np.array(range(0, len(y_test))), y_test, color='b')
    plt.show()
    plt.plot(history.history['loss'])
    plt.show()
