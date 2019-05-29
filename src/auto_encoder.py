import numpy as np
import pandas as pd
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def build_autoencoder(input_size):
    # 预处理数据
    feature_train = pd.read_csv("./dataset/pre_processed/feature_log_train.csv",index_col=0) # index_col
    feature_test = pd.read_csv("./dataset/pre_processed/feature_log_test.csv",index_col=0)
    feature_train = np.array(feature_train)
    feature_test = np.array(feature_test)

    feature_train_data = np.reshape(feature_train,(len(feature_train), 1 ,input_size))
    feature_test_data = np.reshape(feature_test,(len(feature_test), 1 ,input_size))

    input_ = Input(shape=(1,input_size))

    # 编码层
    encoded = Dense(40, activation='relu')(input_)
    encoded = Dense(30, activation='relu')(encoded)
    encoder_output = Dense(20,activation='relu')(encoded)

    # 解码层
    decoded = Dense(30, activation='relu')(encoder_output)
    decoded = Dense(40, activation='relu')(decoded)
    output_ = Dense(input_size, activation='sigmoid')(decoded)

    # 构建自编码模型
    auto_encoder = Model(inputs=input_, outputs=output_)

    # 构建编码模型
    encoder = Model(inputs=input_, outputs=encoder_output)

    auto_encoder.compile(optimizer='adam',loss='mean_squared_error')
    auto_encoder.fit(feature_train_data,feature_train_data, epochs=200)

    print(auto_encoder.evaluate(feature_test_data,feature_test_data))

    log_data = pd.read_csv('./dataset/pre_processed/log_data.csv',index_col=0)
    log_data_with_feature= []
    for i in range(len(log_data)):
        row = np.array(log_data.iloc[i,:])
        values=np.reshape(row,(1,1,55))
        coded = encoder.predict(values)
        coded = np.reshape(coded,(20,))
        log_data_with_feature.append(coded)
    
    log_data_with_feature_ = pd.DataFrame(log_data_with_feature)
    log_data_with_feature_.to_csv('./dataset/after_encoded/log_data_with_feature.csv')

    
if __name__ == "__main__":
    build_autoencoder(55)