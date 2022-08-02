import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np


def train_PR(dataset):
    # maybe normalize data
    train_data = dataset[dataset['Year'] <= 2018]
    result_data = dataset[dataset['Year'] == 2019]
    x_train, x_result, y_train, y_result = train_test_split(train_data, result_data, test_size=0.2)
    model = Sequential()
    # return_sequence = True, if you use second layer
    model.add(LSTM((3, 5), batch_input=(None, None, 5)))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # summary gives the shape and parameters of the model
    # model.summary()
    # history to visualise the loss, epochs the number of fits for the model
    history = model.fit(x_train, y_train, epochs=50, validation_data=(x_result, y_result))
    results = model.predict(x_result)
    plt.scatter(range(len(results)), results, color='r')
    plt.scatter(range(len(y_result)), y_result, color='g')
    plt.show()
    plt.plot(history.history['loss'])
    plt.show()
    # model.score(x_result, y_result)
