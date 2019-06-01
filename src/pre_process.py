import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [12.8, 9.6]

# 小波变换
def wavelet(stock_col):
    output = []
    for i in range((len(stock_col)//10)*10 - 11):
        x = np.array(stock_data.iloc[i: i + 11, 5])
        (ca, cd) = pywt.dwt(x, "haar")
        cat = pywt.threshold(ca, np.std(ca), mode="soft")
        cdt = pywt.threshold(cd, np.std(cd), mode="soft")
        tx = pywt.idwt(cat, cdt, "haar")
        output.append(output, np.mean(tx))
    return output

# 滑动平均
def smooth(stock_col,WSZ):
    out0 = np.convolve(stock_col,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(stock_col[:WSZ-1])[::2]/r
    stop = (np.cumsum(stock_col[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start,out0,stop))

def get_feature_train_dataset(stock_data):
    data_y = []
    log_data_y = []
    log_data = []
    for i in range((len(stock_data)//10)*10 - 11):
        #train = []
        log_ret = []
        for j in range(1, 6): #逐列读数据，一次读11个
            x = np.array(stock_data.iloc[i: i + 11, j])
            (ca, cd) = pywt.dwt(x, "haar")
            cat = pywt.threshold(ca, np.std(ca), mode="soft")
            cdt = pywt.threshold(cd, np.std(cd), mode="soft")
            tx = pywt.idwt(cat, cdt, "haar")

            log = np.diff(np.log(tx))*100 #求两天的差值，并求对数，用变化值
            log_ret = np.append(log_ret, log)

        log_data.append(log_ret)
    log_data = pd.DataFrame(log_data, index=None)
    log_data.to_csv("./dataset/pre_processed/log_data.csv")

    # Close价格
    for i in range((len(stock_data) // 10) * 10 - 11):
        y = 100*np.log(stock_data.iloc[i + 11, 5] / stock_data.iloc[i + 10, 5])
        log_data_y.append(y)
    log_data_y = pd.DataFrame(log_data_y, index=None)
    log_data_y.to_csv("./dataset/pre_processed/log_data_y.csv")

    # 没有log的test
    for i in range((len(stock_data) // 10) * 10 - 11):
        y = stock_data.iloc[i+11, 5]
        data_y.append(y)
    test_data_y = np.array(data_y)[int((1-0.25)*0.8*len(data_y)+0.25*len(data_y)+1):]
    test_data_y = pd.DataFrame(test_data_y,index=None)
    test_data_y.to_csv("./dataset/pre_processed/test_data_y.csv")

    feature_train = pd.DataFrame(log_data[0:int(0.8*0.25*len(log_data))], index=None)
    feature_train.to_csv("./dataset/pre_processed/feature_log_train.csv")

    feature_test = pd.DataFrame(log_data[int(0.8*0.25*len(log_data))+1: int(1.0*0.25*len(log_data))], index=None)
    feature_test.to_csv("./dataset/pre_processed/feature_log_test.csv")

def get_train_dataset():
    with_feature = pd.read_csv("./dataset/after_encoded/log_data_with_feature.csv", index_col=0)

    train_data = np.array(with_feature)[int(0.25*len(with_feature))+1:int(0.25*len(with_feature)+(1-0.25)*0.8*len(with_feature))]
    train_data = pd.DataFrame(train_data,index=None)
    train_data.to_csv("./dataset/train/auto_encoded_train_data.csv")

def get_test_dataset():
    with_feature = pd.read_csv("./dataset/after_encoded/log_data_with_feature.csv", index_col=0)

    test_data = np.array(with_feature)[int((1-0.25)*0.8*len(with_feature)+0.25*len(with_feature)+1):]
    test_data = pd.DataFrame(test_data,index=None)
    test_data.to_csv("./dataset/train/auto_encoded_test_data.csv")

def get_train_dataset_y():
    data_y = pd.read_csv("./dataset/pre_processed/log_data_y.csv", index_col=0)

    train_data_y = np.array(data_y)[int(0.25*len(data_y))+1:int((1-0.25)*0.8*len(data_y)+0.25*len(data_y))]
    train_data_y = pd.DataFrame(train_data_y, index=None)
    train_data_y.to_csv("./dataset/train/train_data_y.csv")

def get_test_dataset_y():
    data_y = pd.read_csv("./dataset/pre_processed/log_data_y.csv", index_col=0)

    test_data_y = np.array(data_y)[int((1-0.25)*0.8*len(data_y)+0.25*len(data_y)+1):]
    test_data_y = pd.DataFrame(test_data_y, index=None)
    test_data_y.to_csv("./dataset/train/test_data_y.csv")    

if __name__ == "__main__":
    stock_data = pd.read_csv("./dataset/stock_data.csv")
    # close = stock_data['Close']
    # date = stock_data['Date']
    # #output = wavelet(close,'test')
    # output = smooth(close, 11)
    # plt.plot(close[:300],label='original')
    # plt.plot(output[:300],label='after moving average')
    # plt.legend()
    
    # plt.savefig('./doc/after_moving_avg.png')
    # plt.show()
    #get_feature_train_dataset(stock_data)
    get_test_dataset()
    get_test_dataset_y()
    get_train_dataset()
    get_train_dataset_y()


