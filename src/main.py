import pre_process as pp
import train
import auto_encoder as ae
import lstm
import get_data as gd

import numpy as np
import pandas as pd

if __name__ == "__main__":
    #得到股票数据
    gd.get_stock_data("AAPL", "2000-01-01", "2019-10-01")

    #数据预处理，得到特征训练集
    stock_data = pd.read_csv("./dataset/stock_data.csv")
    pp.get_feature_train_dataset(stock_data)

    #训练栈式自编码器
    ae.build_autoencoder(55)

    #得到训练数据
    pp.get_train_dataset()
    pp.get_train_dataset_y()
    pp.get_test_dataset()
    pp.get_test_dataset_y()

    #LSTM训练
    train.train(20)