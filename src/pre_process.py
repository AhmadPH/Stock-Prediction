import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

def wavelet(stock_data):
    output = []
    for i in range((len(stock_data)//10)*10 - 11):
        output_1 = []
        for j in range(1, 6):
            x = np.array(stock_data.iloc[i: i + 11, j])
            (ca, cd) = pywt.dwt(x, "haar")
            cat = pywt.threshold(ca, np.std(ca), mode="soft")
            cdt = pywt.threshold(cd, np.std(cd), mode="soft")
            tx = pywt.idwt(cat, cdt, "haar")
            output_1 = np.append(output, tx)   
        output.append(output_1)
    trained = pd.DataFrame(output)
    trained.to_csv("after_wavelet.csv")

def make_wavelet_train(stock_data):
    train_data = []
    test_data = []
    log_train_data = []
    for i in range((len(stock_data)//10)*10 - 11):
        train = []
        log_ret = []
        for j in range(1, 6): #逐列读数据，一次读11个
            x = np.array(stock_data.iloc[i: i + 11, j])
            (ca, cd) = pywt.dwt(x, "haar")
            cat = pywt.threshold(ca, np.std(ca), mode="soft")
            cdt = pywt.threshold(cd, np.std(cd), mode="soft")
            tx = pywt.idwt(cat, cdt, "haar")
            log = np.diff(np.log(tx))*100 #求两天的差值，并求对数
            macd = np.mean(x[5:]) - np.mean(x)
                # ma = np.mean(x)
            sd = np.std(x)
            log_ret = np.append(log_ret, log)
            x_tech = np.append(macd*10, sd)
            train = np.append(train, x_tech)
        train_data.append(train)
        log_train_data.append(log_ret)
    trained = pd.DataFrame(train_data)
    trained.to_csv("./dataset/pre_processed/indicators.csv")
    log_train = pd.DataFrame(log_train_data, index=None)
    log_train.to_csv("./dataset/pre_processed/log_train.csv")
        # auto_train = pd.DataFrame(train_data[0:800])
        # auto_test = pd.DataFrame(train_data[801:1000])
        # auto_train.to_csv("auto_train.csv")
        # auto_test.to_csv("auto_test.csv")
    # rbm_train = pd.DataFrame(log_train_data[0:int(self.split*self.feature_split*len(log_train_data))], index=None)
    # rbm_train.to_csv("./dataset/pre_processed/rbm_train.csv")
    # rbm_test = pd.DataFrame(log_train_data[int(self.split*self.feature_split*len(log_train_data))+1:
    #                                            int(self.feature_split*len(log_train_data))])
    # rbm_test.to_csv("./dataset/pre_processedg/rbm_test.csv")
    # for i in range((len(self.stock_data) // 10) * 10 - 11):
    #     y = 100*np.log(self.stock_data.iloc[i + 11, 5] / self.stock_data.iloc[i + 10, 5])
    #     test_data.append(y)
    # test = pd.DataFrame(test_data)
    # test.to_csv("./dataset/pre_processed/test_data.csv")

if __name__ == "__main__":
    stock_data = pd.read_csv("./dataset/stock_data.csv")
    # stock_data['Close'].plot()
    # plt.title('shoupan')
    # plt.show()
    #make_wavelet_train(stock_data)
    wavelet(stock_data)