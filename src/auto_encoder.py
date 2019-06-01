import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

def tf_autoencoder(input_size):
    X =tf.placeholder(tf.float32,shape=[None,input_size])
    he_init = tf.contrib.layers.variance_scaling_initializer() 
    l2_regularizer = tf.contrib.layers.l2_regularizer(0.001)
    dense_layer = partial(tf.layers.dense, activation=tf.nn.relu,kernel_initializer=he_init,kernel_regularizer=l2_regularizer)

    encoded1 = dense_layer(X, 40)
    encoded2 = dense_layer(encoded1, 30)
    encoded_output = dense_layer(encoded2, 20)

    decoded1 = dense_layer(encoded_output, 30)
    decoded2 = dense_layer(decoded1, 40)
    decoded_output = dense_layer(decoded2, input_size)

    reconstruction_loss = tf.reduce_mean(tf.square(decoded_output - X))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([reconstruction_loss] + reg_losses)


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

    #plot_model(auto_encoder,to_file='./model/auto_encoder.png')
    tb = TensorBoard(log_dir='./model/logs/auto_encoder/',histogram_freq=10,batch_size=11,write_graph=True,write_grads=False, write_images=True,embeddings_freq=0)
    callbacks= [tb]

    auto_encoder.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history=auto_encoder.fit(feature_train_data,feature_train_data, epochs=500, callbacks=callbacks, validation_data=(feature_test_data,feature_test_data))

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left') 
    plt.savefig('./results/auto_encoder/auto_encoder_acc.png')
    plt.show()
    #  summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left') 
    plt.savefig('./results/auto_encoder/auto_encoder_loss.png')
    plt.show()

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
    # np.savetxt("./results/prediction_data",np.array(prediction_data))
    # np.savetxt("./results/test_data_y",test_data_y)
    # np.savetxt("./results/oringal_data_test_y",original_data_test_y)
    # np.savetxt("./results/stock_data",np.array(stock_data))