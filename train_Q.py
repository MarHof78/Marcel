import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


# first with only data from Q. Later with data from PR
def train_Q(dataset):
    """Train the Q dataset

    :param dataset: data from Q dataset
    """
    # train_data = dataset[dataset['Year'] <= 2019]
    # result_data = dataset[dataset['Year'] == 2020]
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    # normalize the data
    X = X / len(X)
    y = y / len(y)
    # random_state to use always the same train data
    # x_train, x_result, y_train, y_result = train_test_split(train_data, result_data, test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = Sequential()
    # return_sequence = True, if you use second layer
    # add second layer
    model.add(LSTM(20, batch_input_shape=(None, None, 1), return_sequences=True))
    model.add(LSTM(1, return_sequences=False))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # summary gives the shape and parameters of the model
    model.summary()
    # history to visualise the loss, epochs the number of fits for the model
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    results = model.predict(x_test)
    # model.score(x_result, y_result)
    plt.scatter(range(1000), results[:1000], color='r')
    plt.show()
    plt.scatter(range(1000), y_test[:1000], color='g')
    plt.show()
    plt.plot(history.history['loss'])
    plt.show()
