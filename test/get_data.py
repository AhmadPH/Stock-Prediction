# import pandas as pd
# import pandas_datareader
# import yfinance as yf
# import matplotlib.pyplot as plt
# import datetime

# #爬取08-18十年数据
# start = datetime.datetime(2008,1,1)
# end = datetime.datetime(2018,1,1)
# #必须要调用这个函数
# yf.pdr_override()
# #股票数据
# stock_data = pandas_datareader.data.get_data_yahoo("SPY",start,end)
# print(stock_data)

# #数据可视化
# stock_data['Close'].plot()
# plt.title('股票每日收盘价')
# plt.show()

import pandas_datareader.data as pdr
import yfinance as yf

yf.pdr_override()

class GetData:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end

    # get stock data
    def get_stock_data(self):
        stock_data = pdr.get_data_yahoo(self.ticker, self.start, self.end)
        stock_data.to_csv("stock_data.csv")

    # get twitter data
    # do your code here!

    # get news data
    # do your code here!


if __name__ == "__main__":
    data = GetData("AAPL", "2000-01-01", "2018-10-01")
    data.get_stock_data()
