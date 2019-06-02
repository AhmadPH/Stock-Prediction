import tensorflow as tf
from keras.models import Model
import keras.layers as kl
import keras as kr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import TensorBoard


def nnmodel():
    train_data = np.array(pd.read_csv("./dataset/train/auto_encoded_train_data.csv", index_col=0))
    train_data = np.reshape(train_data, (len(train_data), 20))
    test_data = np.array(pd.read_csv("./dataset/train/auto_encoded_test_data.csv", index_col=0))
    test_data = np.reshape(test_data, (len(test_data), 20))
    train_data_y = np.array(pd.read_csv("./dataset/train/train_data_y.csv", index_col=0))
    test_data_y = np.array(pd.read_csv("./dataset/train/test_data_y.csv", index_col=0))
    price = np.array(pd.read_csv("./dataset/pre_processed/test_data_y.csv", index_col=0))

    model = kr.models.Sequential()
    model.add(kl.Dense(20, input_dim=20, activation="tanh", activity_regularizer=kr.regularizers.l2(0.05)))
    model.add(kl.Dense(20, activation="tanh", activity_regularizer=kr.regularizers.l2(0.01)))
    model.add(kl.Dense(1))

    model.compile(optimizer="sgd", loss="mean_squared_error",metrics=["accuracy","mse"])    
    #plot_model(model,to_file='./model/nnmodel.png')
    tb = TensorBoard(log_dir='./model/logs/',histogram_freq=10,batch_size=10,write_graph=True,write_grads=False, write_images=True,embeddings_freq=0)
    callbacks= [tb]
    history = model.fit(train_data,train_data_y, epochs=50, callbacks=callbacks, validation_data=(test_data,test_data_y))


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left') 
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left') 
    plt.show()
 
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mean_squared_error')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left') 
    plt.show()

    predicted_data = []
    predicted_price = []
    for i in range(len(test_data_y)):
        prediction = model.predict(np.reshape(test_data[i], (1, 20)))
        predicted_data.append(np.reshape(prediction,(1,)))
        price_pred = np.exp(np.reshape(prediction,(1,)))*price[i]
        predicted_price.append(price_pred[0])
        # print(test_data[i])

    plt.plot(predicted_data,label='predict')
    plt.plot(test_data_y,label='original')
    plt.title('log_prediction result')
    plt.legend()
    plt.show()

    plt.plot(price,label='original')
    plt.plot(predicted_price,label='predict')
    plt.title('prediction result')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    nnmodel()

