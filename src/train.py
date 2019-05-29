import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras.layers as kl
from keras.models import Model
from keras import regularizers

plt.rcParams["figure.figsize"] = [12.8, 9.6]
#from bokeh.plotting import output_file, figure, show

def train(input_size):    
    
    train_data = np.array(pd.read_csv("./dataset/train/auto_encoded_train_data.csv",index_col=0))
    train_data_y = np.array(pd.read_csv("./dataset/train/train_data_y.csv",index_col=0))
    test_data = np.array(pd.read_csv("./dataset/train/auto_encoded_test_data.csv",index_col=0))
    test_data_y = np.array(pd.read_csv("./dataset/train/test_data_y.csv",index_col=0))

    train_data = np.reshape(np.array(train_data),(len(train_data),1,input_size))
    test_data = np.reshape(np.array(test_data),(len(test_data),1,input_size))

    #model
    input_ = kl.Input(shape=(1,input_size))

    LSTM = kl.LSTM(5, input_shape=(1, input_size), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_)
    perc = kl.Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(LSTM)
    LSTM_2 = kl.LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                        dropout=0.2, recurrent_dropout=0.2)(perc)
    output_ = kl.Dense(1, activation="sigmoid", activity_regularizer=regularizers.l2(0.001))(LSTM_2)

    model = Model(input_, output_)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

    #train
    model.fit(train_data,train_data_y, epochs=200)
    print(model.evaluate(test_data,test_data_y))

    original_data_test_y = np.array(pd.read_csv("./dataset/pre_processed/test_data_y.csv",index_col=0))
    stock_data=[]
    prediction_data=[]
    for i in range(len(test_data_y)):
        prediction = (model.predict(np.reshape(test_data[i], (1, 1, input_size))))
        prediction_data.append(np.reshape(prediction, (1,)))
        #prediction_corrected = (prediction_data - np.mean(prediction_data))/np.std(prediction_data)
        stock_price = np.exp(np.reshape(prediction, (1,)))*original_data_test_y[i]
        stock_data.append(stock_price[0])
    stock_data[:] = [i - (float(stock_data[0])-float(original_data_test_y[0])) for i in stock_data]
    # plt.plot(prediction_data,lengend='predict')
    # plt.plot(test_data_y,legend='original')
    np.savetxt("./results/prediction_data",np.array(prediction_data))
    np.savetxt("./results/test_data_y",test_data_y)
    np.savetxt("./results/oringal_data_test_y",original_data_test_y)
    np.savetxt("./results/stock_data",np.array(stock_data))

    plt.plot(original_data_test_y,label='original')
    plt.plot(stock_data,label='predict')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    #train(20)
    original_data_test_y=np.loadtxt("oringal_data_test_y")
    stock_data=np.loadtxt("stock_data")

    plt.plot(original_data_test_y,label='original')
    plt.plot(stock_data,label='predict')
    plt.legend()
    plt.show()
